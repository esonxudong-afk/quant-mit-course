"""
Dynamic Grid Spread Module
===========================
Based on a simplified Avellaneda-Stoikov market-making model,
provides dynamic grid spacing recommendations for daily-frequency grid trading.

Core formula:
    delta = sigma_daily * k_risk * (1 + |q| / Q_max)

    Where:
      sigma_daily = sigma_annual / sqrt(252)
      k_risk      = risk coefficient (default 3.0)
      q           = net position (positive = long)
      Q_max       = maximum inventory layers
      bid         = price * (1 - delta / 2)
      ask         = price * (1 + delta / 2)

Usage:
    ds = DynamicGridSpread(k_risk=3.0, max_inventory=10)
    params = ds.adjust_grid_params(price=100.0, close_prices=prices, inventory=3)
    # => {'bid': 96.44, 'ask': 103.89, 'spread_pct': 0.0744, ...}

CLI:
    python dynamic_spread.py --price 100 --vol 0.30 --inventory 3 --max-inv 10
"""

import math
import argparse
import sys
from typing import Tuple, Dict, List


class DynamicGridSpread:
    """Dynamic grid spread calculator using Avellaneda-Stoikov style model.

    Attributes:
        k_risk: Risk aversion coefficient (higher = wider spreads).
        max_inventory: Maximum absolute position size.
        vol_lookback: Minimum number of price points for vol calculation.
    """

    def __init__(self, k_risk: float = 3.0, max_inventory: int = 10,
                 vol_lookback: int = 20):
        """Initialize the dynamic grid spread calculator.

        Args:
            k_risk: Risk coefficient, default 3.0. Higher values produce
                    wider spreads to account for greater risk aversion.
            max_inventory: Maximum absolute inventory level before the
                           inventory penalty term saturates.
            vol_lookback: Minimum number of close prices required for
                          annualised volatility computation.

        Raises:
            ValueError: If k_risk <= 0, max_inventory <= 0, or vol_lookback < 2.
        """
        if k_risk <= 0:
            raise ValueError(f"k_risk must be positive, got {k_risk}")
        if max_inventory <= 0:
            raise ValueError(f"max_inventory must be positive, got {max_inventory}")
        if vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {vol_lookback}")

        self.k_risk = k_risk
        self.max_inventory = max_inventory
        self.vol_lookback = vol_lookback

    # --------------- Volatility ---------------

    def compute_annual_volatility(self, close_prices: List[float]) -> float:
        """Compute annualised volatility from a series of close prices.

        Uses log-returns: ln(p_i / p_{i-1}) and annualises via sqrt(252).

        Args:
            close_prices: List of closing prices, oldest first.

        Returns:
            Annualised volatility as a decimal (e.g. 0.25 = 25%).

        Raises:
            ValueError: If fewer than vol_lookback prices are provided,
                        or if any price <= 0.
        """
        if len(close_prices) < self.vol_lookback:
            raise ValueError(
                f"Need at least {self.vol_lookback} price points, "
                f"got {len(close_prices)}"
            )

        # Validate all prices are positive
        for i, p in enumerate(close_prices):
            if p <= 0:
                raise ValueError(f"Price at index {i} is <= 0: {p}")

        # Use the most recent vol_lookback observations for volatility estimate
        window = close_prices[-self.vol_lookback:]

        log_returns = []
        for i in range(1, len(window)):
            log_ret = math.log(window[i] / window[i - 1])
            log_returns.append(log_ret)

        # Mean of log returns
        n = len(log_returns)
        mean = sum(log_returns) / n

        # Sample standard deviation (ddof=1)
        sum_sq = sum((r - mean) ** 2 for r in log_returns)
        std = math.sqrt(sum_sq / (n - 1))

        # Annualise: std_daily * sqrt(252)
        annual_vol = std * math.sqrt(252)
        return annual_vol

    # --------------- Spread ---------------

    def optimal_spread(self, price: float, vol_annual: float,
                       inventory: int, max_inventory: int) -> Tuple[float, float, float]:
        """Compute optimal bid/ask prices and spread percentage.

        Core formula:
            delta = sigma_daily * k_risk * (1 + |inv| / max_inv)

            where sigma_daily = vol_annual / sqrt(252)

            bid = price * (1 - delta / 2)
            ask = price * (1 + delta / 2)

        Args:
            price: Current mid-price.
            vol_annual: Annualised volatility (decimal).
            inventory: Current net position (positive = long).
            max_inventory: Maximum absolute inventory for penalty
                           normalisation.

        Returns:
            Tuple[float, float, float]: (bid_price, ask_price, spread_pct)

            spread_pct is delta expressed as a decimal (e.g. 0.05 = 5%).

        Raises:
            ValueError: If price <= 0, vol_annual < 0, or max_inventory <= 0.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if vol_annual < 0:
            raise ValueError(f"vol_annual must be non-negative, got {vol_annual}")
        if max_inventory <= 0:
            raise ValueError(f"max_inventory must be positive, got {max_inventory}")

        # Daily volatility
        sigma_daily = vol_annual / math.sqrt(252)

        # Inventory ratio, clamped to [0, 1]
        inv_ratio = min(abs(inventory) / max_inventory, 1.0)

        # Dynamic spread delta
        delta = sigma_daily * self.k_risk * (1.0 + inv_ratio)

        # Floor: minimum spread of 1% when volatility is essentially zero
        if delta < 0.01:
            delta = 0.01

        bid = price * (1.0 - delta / 2.0)
        ask = price * (1.0 + delta / 2.0)

        # Bid cannot be negative
        bid = max(bid, 0.0)

        return (bid, ask, delta)

    # --------------- Convenience ---------------

    def adjust_grid_params(self, price: float, close_prices: List[float],
                           inventory: int) -> Dict[str, float]:
        """One-shot convenience: compute vol → optimal spread → return params.

        Args:
            price: Current mid-price.
            close_prices: Historical close prices for vol estimation.
            inventory: Current net position.

        Returns:
            Dict with keys:
                - bid (float)
                - ask (float)
                - spread_pct (float) — delta as decimal
                - vol_annual (float)
                - inventory_ratio (float) — |inventory| / max_inventory, capped at 1
        """
        vol_annual = self.compute_annual_volatility(close_prices)
        bid, ask, spread = self.optimal_spread(
            price=price,
            vol_annual=vol_annual,
            inventory=inventory,
            max_inventory=self.max_inventory,
        )
        inv_ratio = min(abs(inventory) / self.max_inventory, 1.0)

        return {
            "bid": bid,
            "ask": ask,
            "spread_pct": spread,
            "vol_annual": vol_annual,
            "inventory_ratio": inv_ratio,
        }

    # --------------- Regime Check ---------------

    def regime_check(self, vol_annual: float) -> str:
        """Classify the market regime based on annualised volatility.

        Thresholds:
            < 0.15  → low_vol   (tighten spreads)
            0.15–0.30 → normal
            0.30–0.50 → high_vol  (widen spreads)
            ≥ 0.50  → crisis     (stop trading or max spread)

        Args:
            vol_annual: Annualised volatility (decimal).

        Returns:
            One of: "low_vol", "normal", "high_vol", "crisis".

        Raises:
            ValueError: If vol_annual < 0.
        """
        if vol_annual < 0:
            raise ValueError(f"vol_annual cannot be negative, got {vol_annual}")

        if vol_annual < 0.15:
            return "low_vol"
        elif vol_annual < 0.30:
            return "normal"
        elif vol_annual < 0.50:
            return "high_vol"
        else:
            return "crisis"


# ===================== CLI =====================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamic Grid Spread — Avellaneda-Stoikov simplified",
    )
    parser.add_argument(
        "--price", type=float, required=True,
        help="Current mid-price",
    )
    parser.add_argument(
        "--vol", type=float, default=None,
        help="Annualised volatility (decimal). If omitted, supply --prices for estimation.",
    )
    parser.add_argument(
        "--inventory", type=int, default=0,
        help="Current net position (default 0)",
    )
    parser.add_argument(
        "--max-inv", type=int, default=10,
        help="Maximum absolute inventory (default 10)",
    )
    parser.add_argument(
        "--k-risk", type=float, default=3.0,
        help="Risk coefficient (default 3.0)",
    )
    parser.add_argument(
        "--prices", type=str, nargs="+", default=None,
        help="Space-separated close prices for vol estimation (oldest first). "
             "Example: --prices 98 99 100 101 102",
    )
    parser.add_argument(
        "--regime-only", action="store_true",
        help="Only output the market regime string.",
    )
    return parser


def main(argv: List[str] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = DynamicGridSpread(
        k_risk=args.k_risk,
        max_inventory=args.max_inv,
    )

    # Determine volatility
    if args.vol is not None:
        vol_annual = args.vol
    elif args.prices is not None:
        prices = [float(p) for p in args.prices]
        vol_annual = ds.compute_annual_volatility(prices)
    else:
        parser.error("Either --vol or --prices must be provided")

    if args.regime_only:
        print(ds.regime_check(vol_annual))
        return

    bid, ask, spread = ds.optimal_spread(
        price=args.price,
        vol_annual=vol_annual,
        inventory=args.inventory,
        max_inventory=args.max_inv,
    )
    regime = ds.regime_check(vol_annual)
    inv_ratio = min(abs(args.inventory) / args.max_inv, 1.0)

    # Pretty-print results
    print(f"Price           = {args.price:.4f}")
    print(f"Vol (annual)    = {vol_annual:.4f} ({vol_annual*100:.2f}%)")
    print(f"Inventory       = {args.inventory} / {args.max_inv} (ratio={inv_ratio:.2f})")
    print(f"Regime          = {regime}")
    print(f"---")
    print(f"Bid             = {bid:.4f}")
    print(f"Ask             = {ask:.4f}")
    print(f"Spread (delta)  = {spread:.6f} ({spread*100:.4f}%)")
    print(f"Spread width    = {ask - bid:.4f}")


if __name__ == "__main__":
    main()
