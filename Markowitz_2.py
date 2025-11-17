"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        returns = self.returns[assets]

        look_ret = 120
        look_vol = self.lookback

        for i in range(max(look_ret, look_vol) + 1, len(self.price)):

            # 1. Momentum：過去 120 天累積報酬
            ret_window = returns.iloc[i - look_ret : i]
            cum_ret = (1 + ret_window).prod() - 1  # Series

            pos_mask = (cum_ret > 0).values.astype(float)

            # 若全部負報酬 → 等權
            if pos_mask.sum() == 0:
                w = np.ones(len(assets)) / len(assets)
                self.portfolio_weights.loc[self.price.index[i], assets] = w
                continue

            # 2. Volatility (inverse vol)
            vol_window = returns.iloc[i - look_vol : i]
            sigma = vol_window.std().values
            sigma = np.where(sigma == 0, 1e-8, sigma)

            inv_sigma = 1 / sigma

            # 只保留正動能者
            inv_sigma = inv_sigma * pos_mask

            # 若變成 0（容錯）→ 等權
            if inv_sigma.sum() == 0:
                w = np.ones(len(assets)) / len(assets)
            else:
                w = inv_sigma / inv_sigma.sum()

            self.portfolio_weights.loc[self.price.index[i], assets] = w

        # 前面資料不足 → 全等權
        eqw = np.ones(len(assets)) / len(assets)
        self.portfolio_weights.iloc[: max(look_ret, look_vol)] = 0
        self.portfolio_weights.loc[
            self.price.index[: max(look_ret, look_vol)], assets
        ] = eqw
        
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
