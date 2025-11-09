# 🧮 Stock Price Optimization using Conjugate Gradient and Fibonacci Methods

## 📘 Description
A quantitative finance project implementing and comparing numerical optimization techniques - **Custom Conjugate Gradient (CG)** and **Fibonacci Search** for weight optimization in linear regression models predicting normalized stock closing prices (AAPL, MSFT, NFLX, etc.).  

This project introduces the `StockPriceOptimizer` class that integrates analytical and iterative solvers, showcasing **efficiency, convergence**, and **predictive accuracy** for financial modeling.

---

## 👨‍💻 Authors
- **Sannidhya Das**  
- **Siddhartha Sen**  
- **Soumyadeep Roy**  
- **Suchibrata Patra**  
- **Dipanjan Chakraborty**  
- **Peyasi Mondal**

---

## 📊 Project Overview
In today’s dynamic financial markets, robust stock price forecasting demands efficient optimization techniques that generalize across volatile conditions.

This project implements a Python-based framework — **`StockPriceOptimizer`** — that fits linear regression models on real NASDAQ stock data using:

1. Analytical Least Squares  
2. Scipy’s Conjugate Gradient (CG)  
3. Custom-built Conjugate Gradient (with analytical step size using Fibonacci Search)

The optimizer minimizes the **Mean Squared Error (MSE)** to find the optimal weight vector `w` for each company’s stock price model and compares accuracy, convergence, and runtime.

---

## 📈 Dataset
**Source:** [Kaggle - Stock Market: Historical Data of Top 10 Companies](https://www.kaggle.com/datasets/khushipitroda/stock-market-historical-data-of-top-10-companies)

**Description:**
- Historical daily data (25,160 records) of 10 major NASDAQ companies:  
  `AAPL`, `MSFT`, `TSLA`, `AMZN`, `META`, `NFLX`, `QCOM`, `CSCO`, `SBUX`, `AMD`  
- Each record includes: **Open**, **High**, **Low**, **Close/Last**, and **Volume**  
- Ideal for regression-based stock trend modeling and optimization analysis.

---

## 🧹 Data Preprocessing
1. **Ticker-level filtering** to isolate single company data.  
2. **Numeric normalization** of OHLCV and target fields.  
3. **Date parsing** and sorting for temporal consistency.  
4. **Missing-value removal** for complete observations.  
5. **Feature engineering:** Added `Days_Since_Start` for time-based learning.  
6. **Chronological train-test split** to avoid data leakage.  
7. **Feature standardization (Z-score scaling)** to stabilize optimization.  
8. **Bias term augmentation** for intercept learning.

---

## ⚙️ Methodology

### 🎯 Objective Function
- **Mean Squared Error (MSE)**: A convex quadratic function ensuring stable convergence.

### 🧠 Techniques Implemented
- **Analytical Least Squares (closed-form)**
- **Scipy’s Conjugate Gradient (iterative)**
- **Custom Conjugate Gradient (CG)**  
  - Initialize residuals and search direction  
  - Compute step size (α) via **Fibonacci Line Search**  
  - Update weights, residuals, and conjugate direction (β)  
  - Iterate until tolerance `1e-6` or `max_iter=1000`  

### 📊 Comparative Analysis
All three optimization approaches achieved **near-identical MSEs**, confirming numerical correctness and robustness of the custom CG implementation.

---

## 📑 Results

| Method                    | Runtime (s) | Final MSE | Iterations |
|----------------------------|-------------|------------|-------------|
| Least Squares              | 0.0059      | 0.3626     | N/A         |
| Scipy Conjugate Gradient   | 0.0897      | 0.3626     | 50          |
| Custom Conjugate Gradient  | 0.2999      | 0.3626     | 150         |

### 🔍 Insights
- **High** and **Low** prices were the strongest positive predictors of the Close price.  
- **Open price** carried a negative coefficient → limited predictive power.  
- **CG methods** demonstrated high scalability and memory efficiency.

### 📂 Generated Outputs
- Predicted test plots for each company (e.g., `AAPL_test_predictions.png`)  
- Comparative error metrics printed during runtime.

---

## 💡 Key Findings
- Analytical and CG solvers converge to **nearly identical solutions**.  
- Custom CG validates algorithmic correctness and interpretability.  
- CG provides **scalability and efficiency** for large-scale finance data.  
- Demonstrated consistent predictive stability across **10 NASDAQ tickers**.

---

## 🚀 Usage

### 🔧 Prerequisites
Install dependencies:
```bash
pip install requirements.txt
```

## 🧭 Instructions
1. Clone the repository
2. Place stock_indexes.csv in the root directory
3. Example usage:
```python
from main import StockPriceOptimizer

optimizer = StockPriceOptimizer('stock_indexes.csv')
optimizer.run_analysis(ticker='MSFT', test_split_date='2022-01-01')
```
5. Output includes:
- Model weights
- Test MSE
- Generated prediction plots for each company

---
## 🗂️ Project Structure
```bash
StockPriceOptimizer-CG-Fibonacci/
│   
├── main.py                  # Core StockPriceOptimizer class
├── code.ipynb               # Runs optimization for all tickers
├── stock_indexes.csv        # Input dataset 
├── outputs/
│   ├── MSFT_test_predictions.png
│   ├── AAPL_test_predictions.png
│   └── ...
├── MDTS4311(2)_GR6.pptx     # Presented pptx file
├── README.md                # This file
└── requirements.txt         # Python dependencies

```

## 📚 Citations
- [US Patent: Applying Fibonacci numbers in financial markets](https://patents.google.com/patent/US6877094B1/en)
- [Conjugate Gradient implementation – Towards Data Science](https://towardsdatascience.com/complete-step-by-step-conjugate-gradient-algorithm-from-scratch-and-its-implementation-for-solving-3a4f9d1b4aa1)
- [ScienceDirect – Mathematical background of CG method](https://www.sciencedirect.com/topics/mathematics/conjugate-gradient-method)
- [NASDAQ Official Historical Data](https://www.nasdaq.com/market-activity/historical-data)

## 🙏 Acknowledgements

This work bridges the gap between theoretical optimization algorithms and their real-world application in quantitative finance. It demonstrates the efficiency, interpretability, and scalability of Conjugate Gradient methods in large-scale predictive modeling.

## 📌 Note
This repository contains our group project for the M.Sc. Data Science (Semester 3) course - MDTS4311 Optimization Techniques.
The project domain (Economics & Finance) and optimization methods were pre-assigned, so our implementation focuses strictly on these methods.
We designed a simple, interpretable pipeline to demonstrate concept-to-code mapping. While the current version has a few limitations, they will be addressed and improved in future iterations.
