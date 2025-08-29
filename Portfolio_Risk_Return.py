# file: portfolio_risk_return.py
"""
Portfolio Risk & Return Analysis (stdin-safe)

This script supports both interactive and non-interactive execution:
- CLI args (preferred in non-interactive/sandboxed envs) via argparse.
- Interactive prompts when stdin is connected (tty).
- Falls back to DEMO mode when input is unavailable and no CLI args supplied.

Modes:
- REAL: Load prices from a local CSV (wide format: Date + ticker columns).
- DEMO: Use synthetic demo data (no external dependencies).
- TEST: Run unit tests (use --mode TEST when running non-interactively).

Key fixes compared to previous version:
- Avoids calling input() in environments where stdin is not available.
- Parses command-line args first; only prompts interactively when a tty is present.
- Keeps unit tests and adds 1 new test to increase coverage.

Requirements: numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import textwrap
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- Constants & Types ------------------------- #
TRADING_DAYS = 252

@dataclass
class PortfolioMetrics:
    expected_return: float
    variance: float
    volatility: float
    sharpe_ratio: float

# ------------------------- Core Utilities ------------------------- #

def parse_weights(raw: str, n_assets: int, auto_normalize: bool = True) -> np.ndarray:
    """Parse comma-separated weights; optionally normalize if not summing to 1.

    Accepts strings like "0.5,0.3,0.2" or lists of numbers joined with commas.
    """
    if isinstance(raw, (list, tuple, np.ndarray)):
        weights = np.array(raw, dtype=float)
    elif isinstance(raw, str):
        try:
            weights = np.array([float(x) for x in raw.split(",")], dtype=float)
        except ValueError as exc:
            raise ValueError("Weights must be numeric, comma-separated.") from exc
    else:
        raise ValueError("Weights must be provided as a comma-separated string or list of numbers.")

    if len(weights) != n_assets:
        raise ValueError(f"Expected {n_assets} weights, got {len(weights)}.")

    total = float(weights.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        if auto_normalize:
            if total == 0:
                raise ValueError("Weights sum to 0; cannot normalize.")
            weights = weights / total
        else:
            raise ValueError("Weights must sum to 1.0.")
    return weights


def load_prices_wide_csv(
    path: str,
    tickers: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load wide CSV with Date column and per-ticker columns. Returns DataFrame indexed by Date.

    Columns are expected to match the tickers provided (case-sensitive match to header names).
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found at '{path}'. Provide a valid path.") from exc

    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    tickers = [t.strip().upper() for t in tickers]
    # Map existing columns to upper-case check without changing original names
    cols_upper = {c.upper(): c for c in df.columns}
    missing = [t for t in tickers if t not in cols_upper]
    if missing:
        raise ValueError(f"CSV missing columns for tickers: {', '.join(missing)}")

    # Reindex with original column names matching upper-case tickers
    actual_cols = [cols_upper[t] for t in tickers]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    if start_date:
        df = df.loc[pd.to_datetime(start_date) : ]
    if end_date:
        df = df.loc[: pd.to_datetime(end_date)]

    df = df[actual_cols].astype(float)

    if df.shape[0] < 2:
        raise ValueError("Not enough rows after filtering dates.")

    # Rename columns to standardized upper-case tickers
    df.columns = tickers
    return df


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple returns (period-to-period) from price levels."""
    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(how="all")
    returns = returns.fillna(0.0)
    return returns


def annualize_stats(returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> Tuple[pd.Series, pd.DataFrame]:
    mean_ann = returns.mean() * periods_per_year
    cov_ann = returns.cov() * periods_per_year
    return mean_ann, cov_ann


def compute_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PortfolioMetrics:
    mean_ann, cov_ann = annualize_stats(returns, periods_per_year)
    w = weights.reshape(-1, 1)

    exp_return = float(np.dot(weights, mean_ann.values))
    variance = float(w.T @ cov_ann.values @ w)
    volatility = float(math.sqrt(max(0.0, variance)))

    sharpe = float("nan") if volatility == 0 else (exp_return - risk_free_rate) / volatility

    return PortfolioMetrics(
        expected_return=exp_return,
        variance=variance,
        volatility=volatility,
        sharpe_ratio=sharpe,
    )


def cumulative_portfolio_growth(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    weighted = (returns + 1.0).cumprod()
    growth = (weighted * weights).sum(axis=1)
    if growth.empty:
        return pd.Series(dtype=float)
    return growth / growth.iloc[0]


def plot_prices(prices: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    prices.plot(ax=plt.gca())
    plt.title("Stock Price History")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_portfolio_growth(growth: pd.Series) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(growth, label="Portfolio")
    plt.title("Cumulative Portfolio Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------- DEMO Data ------------------------- #

def demo_prices(tickers: Optional[List[str]] = None, days: int = 3 * TRADING_DAYS) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    n = len(dates)

    mu = np.array([0.08, 0.10, 0.06]) / TRADING_DAYS
    sigma = np.array([0.15, 0.20, 0.10]) / math.sqrt(TRADING_DAYS)

    prices = {}
    for i, t in enumerate(tickers):
        shocks = rng.normal(mu[i % len(mu)], sigma[i % len(sigma)], size=n)
        series = 100 * np.cumprod(1 + shocks)
        prices[t] = series

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "Date"
    return df

# ------------------------- CLI & Input Handling ------------------------- #

def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Risk & Return Analysis")
    p.add_argument("--mode", choices=["REAL", "DEMO", "TEST"], help="Execution mode")
    p.add_argument("--tickers", help="Comma-separated tickers for REAL or DEMO mode (overrides defaults)")
    p.add_argument("--csv-path", help="Path to wide CSV with Date + ticker columns")
    p.add_argument("--weights", help="Comma-separated weights matching tickers (sum to 1)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (optional)")
    p.add_argument("--end", help="End date YYYY-MM-DD (optional)")
    p.add_argument("--risk-free", type=float, help="Annual risk-free rate, e.g., 0.02")
    p.add_argument("--no-plot", action="store_true", help="Disable plotting (useful in headless environments)")
    p.add_argument("--demo-days", type=int, default=3 * TRADING_DAYS, help="Number of trading days for demo data")
    return p.parse_args(argv)


def interactive_prompt(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, OSError):
        raise RuntimeError("Interactive input is not available in this environment.")

# ------------------------- Main Flow ------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_cli_args(argv)

    interactive = sys.stdin is not None and sys.stdin.isatty()

    print("\nPortfolio Risk & Return Analysis (stdin-safe)\n")

    # Determine mode: CLI arg > interactive prompt > fallback to DEMO
    if args.mode:
        mode = args.mode
    elif interactive:
        m = interactive_prompt("Choose mode [REAL/DEMO/TEST] (default DEMO): ").strip().upper() or "DEMO"
        mode = m
    else:
        mode = "DEMO"
        print("No interactive stdin detected and no --mode provided: defaulting to DEMO mode.")

    if mode == "TEST":
        run_tests()
        return

    # risk-free
    if args.risk_free is not None:
        risk_free = args.risk_free
    elif interactive:
        rf = interactive_prompt("Risk-free rate (annual, e.g., 0.02 for 2%, default 0): ").strip()
        risk_free = float(rf) if rf else 0.0
    else:
        risk_free = 0.0

    prices: pd.DataFrame
    weights: np.ndarray

    if mode == "DEMO":
        # Determine tickers
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            tickers = None

        prices = demo_prices(tickers=tickers, days=args.demo_days)
        tickers = list(prices.columns)
        print(f"Using DEMO tickers: {', '.join(tickers)}")

        default_weights = ",".join([str(round(1.0 / len(tickers), 6)) for _ in tickers])
        raw_weights = None
        if args.weights:
            raw_weights = args.weights
        elif interactive:
            raw_weights = interactive_prompt(
                f"Enter weights for {len(tickers)} tickers {tickers} (sum=1). Press Enter for equal weights [{default_weights}]: "
            ).strip() or default_weights
        else:
            raw_weights = default_weights

        weights = parse_weights(raw_weights, len(tickers), auto_normalize=True)

    elif mode == "REAL":
        # tickers required either from args or interactive
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        elif interactive:
            tickers = [t.strip().upper() for t in interactive_prompt(
                "Enter tickers separated by comma (e.g., AAPL,MSFT,GOOGL): "
            ).split(",") if t.strip()]
        else:
            raise RuntimeError("REAL mode requires --tickers and --csv-path when running non-interactively.")

        csv_path = args.csv_path or (interactive_prompt("Path to CSV file with Date + ticker columns: ").strip() if interactive else None)
        if not csv_path:
            raise RuntimeError("CSV path is required for REAL mode (--csv-path).")

        start_date = args.start
        end_date = args.end
        prices = load_prices_wide_csv(csv_path, tickers, start_date, end_date)

        raw_weights = args.weights
        if not raw_weights and interactive:
            raw_weights = interactive_prompt(f"Enter weights for {len(tickers)} tickers {tickers} (sum=1): ").strip()
        if not raw_weights:
            raise RuntimeError("Weights are required for REAL mode when not provided via --weights.")

        weights = parse_weights(raw_weights, len(tickers), auto_normalize=True)

    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    # Compute analytics
    returns = to_returns(prices)
    metrics = compute_metrics(returns, weights, risk_free)

    print("\nPortfolio Analysis Results:")
    print(f"Expected Annual Return: {metrics.expected_return:.6f}")
    print(f"Annual Volatility (Risk): {metrics.volatility:.6f}")
    print(f"Portfolio Variance:      {metrics.variance:.6f}")
    print(f"Sharpe Ratio:            {metrics.sharpe_ratio:.6f}")

    # Visualizations
    if not args.no_plot:
        try:
            plot_prices(prices)
            growth = cumulative_portfolio_growth(returns, weights)
            if not growth.empty:
                plot_portfolio_growth(growth)
        except Exception as plot_exc:
            print(f"Plotting skipped due to: {plot_exc}")
    else:
        print("Plotting disabled (--no-plot).")

# ------------------------- Unit Tests ------------------------- #

def run_tests() -> None:
    import unittest

    class PortfolioTests(unittest.TestCase):
        def test_parse_weights_normalizes(self):
            w = parse_weights("2,2,2", 3, auto_normalize=True)
            self.assertTrue(np.isclose(w.sum(), 1.0))
            np.testing.assert_allclose(w, np.array([1/3, 1/3, 1/3]), rtol=1e-6)

        def test_parse_weights_errors(self):
            with self.assertRaises(ValueError):
                parse_weights("0.5,0.5", 3)
            with self.assertRaises(ValueError):
                parse_weights("a,b,c", 3)
            with self.assertRaises(ValueError):
                parse_weights("0,0,0", 3)

        def test_metrics_known_values(self):
            dates = pd.bdate_range("2023-01-02", periods=TRADING_DAYS)
            r1 = pd.Series(0.001, index=dates)
            r2 = pd.Series(0.002, index=dates)
            rets = pd.concat([r1, r2], axis=1)
            rets.columns = ["A1", "A2"]

            w = np.array([0.6, 0.4])
            mean_ann, cov_ann = annualize_stats(rets)

            exp_mean = float(np.dot(w, mean_ann.values))
            metrics = compute_metrics(rets, w, risk_free_rate=0.0)

            self.assertTrue(np.isclose(metrics.expected_return, exp_mean, rtol=1e-12))
            self.assertTrue(np.isclose(metrics.variance, 0.0, atol=1e-10))
            self.assertTrue(np.isnan(metrics.sharpe_ratio))

        def test_demo_prices_shape(self):
            df = demo_prices(["X", "Y"], days=252)
            self.assertEqual(df.shape[1], 2)
            self.assertGreater(df.shape[0], 10)
            self.assertIn("X", df.columns)

        # NEW TEST: ensure to_returns handles NaNs and fills with zeros
        def test_to_returns_handles_nans(self):
            dates = pd.bdate_range("2024-01-01", periods=5)
            prices = pd.DataFrame({"A": [100.0, None, 102.0, None, 104.0], "B": [50.0, 51.0, None, 52.0, None]}, index=dates)
            rets = to_returns(prices)
            # No NaNs remain
            self.assertFalse(rets.isna().values.any())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PortfolioTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
