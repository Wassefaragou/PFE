import math
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from futures_pnl.analytics import (
    build_cmp_portfolio_view,
    compute_cmp_sequential,
    compute_contract_metrics,
    compute_global_metrics,
)
from futures_pnl.pricing import compute_theoretical_prices
from futures_pnl.validators import validate_contracts, validate_transactions


SETTINGS = {
    "default_tick_value": 10,
    "commission_bvc_rt": 0,
    "commission_broker_rt": 0,
    "commission_sgmat_rt": 0,
    "default_position_limit_per_contract": 100,
}


def _contracts_frame(settlement_price_points: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contract_code": "FMASI20AVR26",
                "underlying_name": "MASI20",
                "expiry_date": "2026-04-30",
                "tick_value": None,
                "initial_margin_per_lot": 1000,
                "settlement_price_points": settlement_price_points,
                "position_limit_per_contract": None,
                "comments": None,
            }
        ]
    )


def _transactions_frame(rows: list[tuple]) -> pd.DataFrame:
    transactions = pd.DataFrame(
        rows,
        columns=[
            "execution_id",
            "trade_date",
            "trade_time",
            "contract",
            "side_lp",
            "quantity_lots",
            "price_points",
        ],
    )
    transactions["counterparty"] = "X"
    transactions["counterparty_type"] = "B"
    transactions["status"] = "CONFIRME"
    return transactions[
        [
            "execution_id",
            "trade_date",
            "trade_time",
            "contract",
            "side_lp",
            "quantity_lots",
            "price_points",
            "counterparty",
            "counterparty_type",
            "status",
        ]
    ]


def _validated_inputs(settlement_price_points: float, trades: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    contracts_validated, contract_issues = validate_contracts(_contracts_frame(settlement_price_points), SETTINGS)
    if not contract_issues.empty:
        raise AssertionError(contract_issues.to_dict(orient="records"))

    contracts_priced = compute_theoretical_prices(contracts_validated, SETTINGS)
    transactions_validated, transaction_issues = validate_transactions(_transactions_frame(trades), contracts_priced)
    if not transaction_issues.empty:
        raise AssertionError(transaction_issues.to_dict(orient="records"))

    return contracts_priced, transactions_validated


class AnalyticsTests(unittest.TestCase):
    def test_wap_and_sequential_methods_keep_same_total_pnl(self) -> None:
        contracts_priced, transactions_validated = _validated_inputs(
            130,
            [
                ("T1", "2026-04-01", "09:00:00", "FMASI20AVR26", "BUY", 1, 100),
                ("T2", "2026-04-01", "10:00:00", "FMASI20AVR26", "SELL", 1, 110),
                ("T3", "2026-04-01", "11:00:00", "FMASI20AVR26", "BUY", 1, 120),
            ],
        )

        contract_metrics = compute_contract_metrics(contracts_priced, transactions_validated, SETTINGS)
        _, cmp_summary = compute_cmp_sequential(transactions_validated, contract_metrics)

        wap_row = contract_metrics.iloc[0]
        sequential_row = cmp_summary.iloc[0]

        self.assertAlmostEqual(wap_row["entry_wap"], 110.0)
        self.assertAlmostEqual(wap_row["pnl_realized_mad"], 0.0)
        self.assertAlmostEqual(wap_row["pnl_unrealized_mad"], 200.0)

        self.assertAlmostEqual(sequential_row["cmp_final_cost"], 120.0)
        self.assertAlmostEqual(sequential_row["cmp_realized_total"], 100.0)
        self.assertAlmostEqual(sequential_row["cmp_unrealized"], 100.0)

        self.assertAlmostEqual(wap_row["pnl_accounting_mad"], sequential_row["cmp_total"])

    def test_capital_engaged_uses_peak_position(self) -> None:
        contracts_priced, transactions_validated = _validated_inputs(
            125,
            [
                ("T1", "2026-04-01", "09:00:00", "FMASI20AVR26", "BUY", 2, 100),
                ("T2", "2026-04-01", "10:00:00", "FMASI20AVR26", "SELL", 1, 110),
                ("T3", "2026-04-01", "11:00:00", "FMASI20AVR26", "BUY", 1, 120),
            ],
        )

        contract_metrics = compute_contract_metrics(contracts_priced, transactions_validated, SETTINGS)
        row = contract_metrics.iloc[0]

        self.assertAlmostEqual(row["abs_position"], 2.0)
        self.assertAlmostEqual(row["peak_abs_position"], 2.0)
        self.assertAlmostEqual(row["capital_engaged_mad"], 2000.0)

    def test_flat_portfolio_keeps_capital_roi_and_hides_open_margin_roi(self) -> None:
        contracts_priced, transactions_validated = _validated_inputs(
            110,
            [
                ("T1", "2026-04-01", "09:00:00", "FMASI20AVR26", "BUY", 1, 100),
                ("T2", "2026-04-01", "10:00:00", "FMASI20AVR26", "SELL", 1, 110),
            ],
        )

        contract_metrics = compute_contract_metrics(contracts_priced, transactions_validated, SETTINGS)
        global_metrics = compute_global_metrics(contract_metrics)

        self.assertAlmostEqual(global_metrics["total_management_pnl"], 100.0)
        self.assertAlmostEqual(global_metrics["total_margin"], 0.0)
        self.assertAlmostEqual(global_metrics["capital_total_engaged"], 1000.0)
        self.assertTrue(math.isnan(global_metrics["roi_on_margin"]))
        self.assertAlmostEqual(global_metrics["roi_on_capital_engaged"], 0.1)

    def test_cmp_portfolio_defaults_missing_tolerance_to_true_without_future_warning(self) -> None:
        contracts_priced, transactions_validated = _validated_inputs(
            110,
            [
                ("T1", "2026-04-01", "09:00:00", "FMASI20AVR26", "BUY", 1, 100),
            ],
        )

        contract_metrics = compute_contract_metrics(contracts_priced, transactions_validated, SETTINGS)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cmp_portfolio = build_cmp_portfolio_view(contract_metrics, pd.DataFrame())

        self.assertTrue(bool(cmp_portfolio.iloc[0]["within_tolerance"]))
        future_warnings = [warning for warning in caught if issubclass(warning.category, FutureWarning)]
        self.assertEqual(future_warnings, [])


if __name__ == "__main__":
    unittest.main()
