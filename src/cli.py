"""Command-line interface for prices, backtests, reports, and an offline demo."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from . import Backtest, MarketData, PerformanceReport, TransactionCostModel
from .backtest import Calendar
from .exceptions import OpenQuantError
from .version import __version__


def _dates(parser):
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)


def _write_csv(frame, output):
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
        print(f"Saved {path}")
    else:
        print(frame.to_csv(), end="")


def _parser():
    parser = argparse.ArgumentParser(prog="openquant", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    yahoo = commands.add_parser("yahoo").add_subparsers(dest="action", required=True)
    prices = yahoo.add_parser("prices", help="Download Yahoo close prices")
    prices.add_argument("tickers", nargs="+")
    _dates(prices)
    prices.add_argument("--output")

    backtest = commands.add_parser("backtest").add_subparsers(dest="action", required=True)
    target = backtest.add_parser("target-weights", help="Backtest close-executed target weights")
    target.add_argument("--tickers", nargs="+", required=True)
    target.add_argument("--weights", nargs="+", type=float, required=True)
    target.add_argument("--rebalance", default="M")
    target.add_argument("--commission-bps", type=float, default=0.0)
    target.add_argument("--slippage-bps", type=float, default=0.0)
    target.add_argument("--initial-cash", type=float, default=1_000_000.0)
    target.add_argument("--output")
    _dates(target)

    reports = commands.add_parser("report").add_subparsers(dest="action", required=True)
    tear = reports.add_parser("tearsheet", help="Generate a single-ticker HTML report")
    tear.add_argument("--ticker", required=True)
    tear.add_argument("--output", default="reports/tear_sheet.html")
    _dates(tear)
    demo = commands.add_parser("demo", help="Run a deterministic synthetic-data demo (offline)")
    demo.add_argument("--out-dir", default="reports")
    return parser


def _demo(out_dir):
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    rng = np.random.default_rng(42)
    px = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (len(dates), 2)), axis=0),
        index=dates,
        columns=["A", "B"],
    )

    class DemoData:
        def prices(self, tickers, **kwargs):
            return px[tickers].copy()

    result = Backtest(
        DemoData(),
        TransactionCostModel(commission_bps=1, slippage_bps=2),
        calendar=Calendar.WEEKDAYS,
    ).run_target_weights({"A": 0.6, "B": 0.4}, "2023-01-02", "2023-12-30")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    result.performance.to_csv(dest / "performance.csv")
    result.holdings.to_csv(dest / "holdings.csv")
    PerformanceReport.from_backtest(result, "Synthetic demo (not market data)").to_html(
        str(dest / "tear_sheet.html")
    )
    print(f"Synthetic demo completed: {dest / 'tear_sheet.html'}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            _demo(args.out_dir)
        elif args.command == "yahoo":
            _write_csv(
                MarketData().prices(args.tickers, start=args.start, end=args.end), args.output
            )
        elif args.command == "backtest":
            if len(args.tickers) != len(args.weights) or len(set(args.tickers)) != len(
                args.tickers
            ):
                parser.error("provide one weight per unique ticker")
            result = Backtest(
                MarketData(),
                TransactionCostModel(args.commission_bps, args.slippage_bps),
                rebalance=args.rebalance,
                initial_cash=args.initial_cash,
            ).run_target_weights(dict(zip(args.tickers, args.weights)), args.start, args.end)
            _write_csv(result.performance, args.output)
        elif args.command == "report":
            result = Backtest(MarketData()).run_target_weights(
                {args.ticker: 1.0}, args.start, args.end
            )
            PerformanceReport.from_backtest(result, args.ticker).to_html(args.output)
            print(f"Saved {args.output}")
    except (OpenQuantError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
