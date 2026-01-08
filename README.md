# 📐 Advanced Monte Carlo Pricing of European Options

## 📌 Project Overview

This project implements a **fully-fledged Monte Carlo pricing engine for European options**, grounded in **stochastic calculus and risk-neutral valuation theory**.

It combines:

* **Analytical Black–Scholes pricing**
* **Monte Carlo simulation under GBM**
* **Advanced variance reduction techniques**
* **Monte Carlo estimation of Greeks**
* **Convergence and efficiency analysis**
* **Clear theoretical documentation**

The project is designed to mirror **quantitative finance coursework and real-world quant research workflows**.

---

## 🎯 Objectives

* Price European options under the **risk-neutral measure**
* Compare **Monte Carlo estimates** against **closed-form Black–Scholes prices**
* Reduce simulation error using **variance reduction techniques**
* Estimate **Greeks using Monte Carlo methods**
* Analyze **convergence behavior and computational efficiency**
* Connect **stochastic calculus theory** with numerical implementation

---

## 🧠 Core Quantitative Concepts

* Risk-neutral valuation
* Geometric Brownian Motion (GBM)
* Ito’s Lemma
* Girsanov’s Theorem
* Monte Carlo simulation
* Variance reduction
* Greeks estimation
* Numerical convergence analysis

---

## 🛠️ Technologies & Libraries

* **Python**
* **NumPy** – numerical computation
* **SciPy** – probability distributions
* **Matplotlib & Seaborn** – visualization
* **Object-Oriented Programming (OOP)**

---

## 📂 Project Structure

*All files are located in a single directory.*

```
Option-Pricing-Monte-Carlo/
│
├── black_scholes.py                 # Analytical Black–Scholes formulas
├── monte_carlo_pricer.py            # Monte Carlo pricing engine
├── variance_reduction_analysis.py   # Variance reduction comparison
├── utils.py                         # Visualization utilities
├── main.py                          # End-to-end execution script
├── stochastic_calculus.md           # Theory & mathematical foundations
└── README.md
```

---

## 📘 Theoretical Foundation

The pricing framework is derived rigorously using **stochastic calculus**.

### Risk-Neutral Dynamics

Under the risk-neutral measure ( \mathbb{Q} ):
[
dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}
]

### Exact GBM Solution

[
S_{t+\Delta t} = S_t \exp\left((r - \frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\right)
]

The full derivation using **Girsanov’s Theorem** and **Ito’s Lemma** is documented in
📄 `stochastic_calculus.md`.

---

## 📈 Pricing Methods Implemented

### 1️⃣ Analytical Black–Scholes

* Closed-form pricing for European calls and puts
* Exact Greeks:

  * Delta, Gamma, Vega, Theta, Rho

### 2️⃣ Monte Carlo Pricing

* Risk-neutral GBM simulation
* Discounted payoff estimation
* Standard error computation

---

## ⚡ Variance Reduction Techniques

To improve efficiency and reduce estimator variance, the following are implemented:

* **Antithetic Variates**
* **Control Variates**
* **Moment Matching**

Each method is:

* Benchmarked against basic Monte Carlo
* Compared using variance and standard error
* Evaluated using efficiency gains

---

## 📐 Monte Carlo Greeks Estimation

Greeks are computed numerically using **bump-and-revalue with path recycling**:

* Delta
* Gamma
* Vega
* Theta
* Rho

Results are directly compared to analytical Black–Scholes Greeks, including absolute errors.

---

## 📊 Visualization & Analysis

The project generates:

* Monte Carlo price convergence plots
* Error decay vs ( O(1/\sqrt{N}) )
* Payoff and discounted payoff distributions
* Sample GBM paths
* Variance and efficiency comparison charts

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install numpy scipy matplotlib seaborn
```

### 2️⃣ Run the Main Script

```bash
python main.py
```

This will:

* Price options using multiple Monte Carlo methods
* Compare results to Black–Scholes
* Compute Greeks
* Generate convergence & variance reduction plots

---

## 📚 What I Learned

* Translating **stochastic calculus theory into numerical code**
* Practical challenges of Monte Carlo simulation
* Importance of variance reduction in computational finance
* Numerical estimation of Greeks
* Interpreting convergence and efficiency trade-offs
