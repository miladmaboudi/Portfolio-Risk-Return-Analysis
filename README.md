# Portfolio Risk & Return Analysis

This Python tool analyzes the **risk and return profile** of an investment portfolio.  
It works in both **interactive** and **non-interactive** environments, making it suitable for local use, automation, and headless servers.

---

## Features
- Supports **REAL** mode (with local CSV data) and **DEMO** mode (synthetic data).  
- **TEST** mode runs unit tests for validation.  
- CLI-safe: avoids crashes when `stdin` is unavailable.  
- Computes:
  - Expected Annual Return  
  - Annual Volatility (Risk)  
  - Portfolio Variance  
  - Sharpe Ratio  
- Optional **plotting** of price history & cumulative growth.  
- **Unit tests included** (`--mode TEST`).  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-risk-return.git
cd portfolio-risk-return
pip install -r requirements.txt
