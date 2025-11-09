import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import traceback

class StockPriceOptimizer:
    """
    A class to optimize linear regression weights for stock price prediction
    using various methods (Scipy CG, Custom CG, Least Squares).
    """
    
    def __init__(self, csv_file_path):
        """
        Initializes the optimizer by loading the full dataset.
        
        Args:
            csv_file_path (str): Path to the stock indexes CSV file.
        """
        try:
            self.full_data = pd.read_csv(csv_file_path)
            print(f"Successfully loaded data from {csv_file_path}")
        except FileNotFoundError:
            print(f"Error: The file '{csv_file_path}' was not found.")
            self.full_data = pd.DataFrame()

    # --- 1. Main Public Method ---
    
    def run_analysis(self, ticker, test_split_date, 
                       feature_cols=['Volume', 'Open', 'High', 'Low'], 
                       target_col='Close/Last'):
        """
        Runs the full analysis for a specific stock ticker.
        """
        if self.full_data.empty:
            print("Cannot run analysis: Data was not loaded successfully.")
            return

        print(f"\n--- Starting Analysis for: {ticker} ---")
        
        try:
            # Use .copy() to avoid SettingWithCopyWarning
            ticker_data = self.full_data[self.full_data['Company'] == ticker].copy()
            if ticker_data.empty:
                print(f"Error: No data found for ticker '{ticker}'.")
                return
                
            clean_data = self._preprocess_data(ticker_data, feature_cols, target_col)
            
            train_df, test_df = self._time_based_split(clean_data, test_split_date)
            
            if train_df.empty or test_df.empty:
                print(f"Error: Split date '{test_split_date}' results in empty train or test set.")
                return

            print(f"Data split: {len(train_df)} train samples, {len(test_df)} test samples.")
            
            X_train_raw, y_train = self._get_features_target(train_df, feature_cols)
            X_test_raw, y_test = self._get_features_target(test_df, feature_cols)
            
            X_train, X_test = self._standardize_features(X_train_raw, X_test_raw)
            
            all_weights = self._fit_models(X_train, y_train)
            
            results, predictions = self._evaluate_models(all_weights, X_test, y_test)
            
            print("\n--- Model Test Set Results ---")
            for method, result in results.items():
                print(f"Method: {method}")
                # Use np.nan_to_num to print 'nan' or 'inf' cleanly
                print(f"  Test MSE: {np.nan_to_num(result['mse']):.6f}")
                print(f"  Weights: {np.nan_to_num(result['weights'])}")

            self._plot_results(test_df['Date'], y_test, predictions, ticker, results)
            
            print(f"{ticker} analysis completed.")

        except Exception as e:
            print(f"An unexpected error occurred during analysis for {ticker}: {e}")
            traceback.print_exc()

    # --- 2. Internal Data Processing Methods ---

    def _preprocess_data(self, data, feature_cols, target_col):
        """Cleans and feature-engineers the data for a single ticker."""
        cols_to_convert = feature_cols + [target_col]
        
        for col in cols_to_convert:
            # .loc is the correct way, but we are operating on a copy `data`
            data[col] = data[col].astype(str).str.replace(r'[$,]', '', regex=True)
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        all_cols = ['Date'] + cols_to_convert
        # This .copy() ensures `data` is a clean DataFrame
        data = data.dropna(subset=all_cols).copy()
        
        # This assignment is on the clean copy, which is safe.
        data.loc[:, 'Days_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days
        return data

    def _time_based_split(self, data, split_date):
        """Splits the data based on a date."""
        split_datetime = pd.to_datetime(split_date)
        train_df = data[data['Date'] < split_datetime].copy()
        test_df = data[data['Date'] >= split_datetime].copy()
        return train_df, test_df

    def _get_features_target(self, df, feature_cols):
        """Extracts X and y matrices from a DataFrame."""
        feature_cols_with_time = ['Days_Since_Start'] + feature_cols
        X = df[feature_cols_with_time].values
        y = df['Close/Last'].values
        return X, y

    def _standardize_features(self, X_train, X_test):
        """Standardizes features based on *training* data mean/std."""
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        
        self.std[self.std == 0] = 1.0 
        
        X_train_scaled = (X_train - self.mean) / self.std
        X_test_scaled = (X_test - self.mean) / self.std
        
        X_train_final = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
        X_test_final = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]
        
        return X_train_final, X_test_final

    # --- 3. Model Fitting Methods (Train Data) ---

    def _fit_models(self, X_train, y_train):
        """Trains all three models on the training data."""
        initial_weights = np.zeros(X_train.shape[1])
        all_weights = {}

        print("\nFitting models...")
        
        print("  Fitting Scipy CG...")
        result_scipy = minimize(self._objective_function, 
                                initial_weights, 
                                args=(X_train, y_train), 
                                method='CG',
                                jac=self._gradient_function)
        all_weights['Scipy CG'] = result_scipy.x

        # THIS IS THE CORRECT, STABLE VERSION
        print("  Fitting Custom CG (with Analytical Alpha)...") 
        weights_custom, _, _ = self._conjugate_gradient_custom(
            self._objective_function, 
            self._gradient_function, 
            initial_weights, 
            X_train, 
            y_train
        )
        all_weights['Custom CG'] = weights_custom

        print("  Fitting Least Squares...")
        all_weights['Least Squares'] = self._least_squares(X_train, y_train)
        
        print("All models fitted.")
        return all_weights

    # --- 4. Model Evaluation Methods (Test Data) ---

    def _evaluate_models(self, all_weights, X_test, y_test):
        """Calculates MSE and predictions for all models on test data."""
        results = {}
        predictions = {}
        
        for method, weights in all_weights.items():
            pred_test = X_test @ weights
            mse_test = self._objective_function(weights, X_test, y_test)
            
            predictions[method] = pred_test
            results[method] = {'mse': mse_test, 'weights': weights}
            
        return results, predictions

    # --- 5. Optimization & Math Helper Methods ---

    @staticmethod
    def _objective_function(weights, X, y):
        """Static method for Mean Squared Error."""
        if not np.isfinite(weights).all():
            return np.inf
        predictions = X @ weights
        error = np.mean((predictions - y) ** 2)
        if not np.isfinite(error):
            return np.inf
        return error

    @staticmethod
    def _gradient_function(weights, X, y):
        """Static method for MSE gradient."""
        if not np.isfinite(weights).all():
            return np.full_like(weights, np.inf)
        predictions = X @ weights
        if not np.isfinite(predictions).all():
            return np.full_like(weights, np.inf)
        gradient = 2 * X.T @ (predictions - y) / len(y)
        return gradient

    @staticmethod
    def _least_squares(X, y):
        """Static method for Least Squares (Normal Equation)."""
        try:
            weights = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            print("Warning: Least Squares matrix is singular. Using pseudo-inverse.")
            weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        return weights

    # ### THIS IS THE CORRECTED, STABLE FUNCTION ###
    def _conjugate_gradient_custom(self, f, grad_f, x0, X, y, tol=1e-6, max_iter=1000):
        """
        Custom CG implementation with analytical step size (alpha)
        and Fletcher-Reeves (beta) for numerical stability.
        
        This version DOES NOT use Fibonacci search. It uses the exact
        mathematical formula for the step size, which is stable.
        """
        x = x0
        g = grad_f(x, X, y)  # Initial gradient
        r = -g               # Initial residual (negative gradient)
        d = r                # Initial search direction
        history = [f(x, X, y)]
        iteration_count = 0
        
        if np.allclose(g, 0.0):
            return x, history, 0

        r_dot_r = np.dot(r, r) # Pre-calculate dot product

        for i in range(max_iter):
            iteration_count += 1
            
            # H = (2/N) * X.T @ X
            # H_d = H @ d = (2/N) * X.T @ (X @ d)
            H_d_vec = (2 / len(y)) * X.T @ (X @ d)
            
            # alpha_den = d.T @ H @ d
            alpha_den = np.dot(d, H_d_vec)
            
            if alpha_den <= 1e-10:
                break 

            alpha = r_dot_r / alpha_den
            
            if not np.isfinite(alpha):
                break

            x = x + alpha * d
            
            # Update residual 'r' efficiently
            r_next = r - alpha * H_d_vec
            
            # Convergence check
            r_next_dot_r_next = np.dot(r_next, r_next)
            if np.sqrt(r_next_dot_r_next) < tol:
                history.append(f(x, X, y))
                x = x + alpha * d # Final update
                break

            beta_num = r_next_dot_r_next
            beta_den = r_dot_r
            
            if np.abs(beta_den) < 1e-10:
                break # Denominator is zero
            
            beta = beta_num / beta_den
            
            d = r_next + beta * d
            r = r_next
            r_dot_r = r_next_dot_r_next # Update for next loop
            
            history.append(f(x, X, y))
            
            if not np.isfinite(x).all():
                break # Safety check for overflow

        return x, history, iteration_count

    # --- 6. Plotting Method ---

    def _plot_results(self, test_dates, y_test, predictions, ticker, results):
        """Plots actual vs. predicted prices on the test set."""
        print(f"Generating plot for {ticker}...")
        plt.figure(figsize=(14, 8))
        
        # 1. Plot the actual price first, as the base layer
        plt.plot(test_dates, y_test, label='Actual Close Price', color='red', linewidth=1, zorder=10)
        
        other_styles = {
            'Scipy CG': {'color': 'green', 'linestyle': '--', 'zorder': 11},
            'Least Squares': {'color': 'black', 'linestyle': '-.', 'zorder': 11}
        }
        
        for method, pred in predictions.items():
            # Don't plot if predictions are invalid
            if not np.isfinite(pred).all():
                continue
                
            mse = results[method]['mse']
            
            if method == 'Custom CG':
                
                plt.plot(test_dates, pred, 
                         label=f"Predicted ({method}) - MSE: {mse:.4f}",
                         color='blue',           
                         linestyle='-',         
                         linewidth=1.0,         
                         zorder=12)             # Highest z-order (draw on top)
            else:
                # Plot other methods 
                style = other_styles.get(method, {'color': 'gray', 'linestyle': ':', 'zorder': 11})
                plt.plot(test_dates, pred, 
                         label=f"Predicted ({method}) - MSE: {mse:.4f}", 
                         color=style['color'],
                         linestyle=style['linestyle'],
                         linewidth=1.0,         # Make other lines thinner
                         zorder=style['zorder'],
                         alpha=0.9)

        plt.title(f'{ticker} Stock Price: Test Set Predictions')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f'{ticker}_test_predictions.png'
        plt.savefig(plot_filename)
        print(f"Plot saved as '{plot_filename}'")
