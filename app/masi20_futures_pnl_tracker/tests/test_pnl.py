import runpy
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PNL_APP_ROOT = ROOT / "app" / "masi20_futures_pnl_tracker"
if str(PNL_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(PNL_APP_ROOT))

import streamlit as st

import futures_pnl.ui as ui
from futures_pnl.analytics import (
    build_cmp_portfolio_view,
    compute_cmp_global_metrics,
    compute_cmp_sequential,
    compute_confirmed_positions,
    compute_contract_metrics,
    compute_global_metrics,
)


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text_input(self, *args, **kwargs):
        return ""

    def date_input(self, *args, **kwargs):
        return pd.Timestamp("2026-04-20").date()

    def time_input(self, *args, **kwargs):
        return pd.Timestamp("2026-04-20 09:00:00").time()

    def selectbox(self, label, options, **kwargs):
        return options[0] if options else None

    def number_input(self, *args, **kwargs):
        return 0.0


def build_sample_state() -> dict:
    contracts = pd.DataFrame(
        [
            {
                "contract_code": "M2L",
                "underlying_name": "Long Case",
                "expiry_date": "2026-06-30",
                "days_to_expiry": 68,
                "expiry_alert": "OK",
                "settlement_price_points": 105.0,
                "mtm_price": 105.0,
                "mtm_source": "contract",
                "effective_tick_value": 10.0,
                "effective_initial_margin_per_lot": 500.0,
                "effective_position_limit_per_contract": 20.0,
                "is_valid": True,
            },
            {
                "contract_code": "M2S",
                "underlying_name": "Short Case",
                "expiry_date": "2026-06-30",
                "days_to_expiry": 68,
                "expiry_alert": "OK",
                "settlement_price_points": 190.0,
                "mtm_price": 190.0,
                "mtm_source": "contract",
                "effective_tick_value": 5.0,
                "effective_initial_margin_per_lot": 400.0,
                "effective_position_limit_per_contract": 20.0,
                "is_valid": True,
            },
        ]
    )

    transactions = pd.DataFrame(
        [
            {
                "execution_id": "T1",
                "trade_date": "2026-04-20",
                "trade_time": "09:00:00",
                "contract": "M2L",
                "side_lp": "BUY",
                "quantity_lots": 3.0,
                "price_points": 100.0,
                "counterparty": "X",
                "counterparty_type": "BROKER",
                "status": "CONFIRMED",
                "is_official_for_calc": True,
                "is_confirmed": True,
                "is_valid_for_calc": True,
                "chrono_key": 1,
            },
            {
                "execution_id": "T2",
                "trade_date": "2026-04-20",
                "trade_time": "10:00:00",
                "contract": "M2L",
                "side_lp": "SELL",
                "quantity_lots": 1.0,
                "price_points": 110.0,
                "counterparty": "X",
                "counterparty_type": "BROKER",
                "status": "CONFIRMED",
                "is_official_for_calc": True,
                "is_confirmed": True,
                "is_valid_for_calc": True,
                "chrono_key": 2,
            },
            {
                "execution_id": "T3",
                "trade_date": "2026-04-20",
                "trade_time": "11:00:00",
                "contract": "M2S",
                "side_lp": "SELL",
                "quantity_lots": 4.0,
                "price_points": 200.0,
                "counterparty": "Y",
                "counterparty_type": "BROKER",
                "status": "CONFIRMED",
                "is_official_for_calc": True,
                "is_confirmed": True,
                "is_valid_for_calc": True,
                "chrono_key": 3,
            },
            {
                "execution_id": "T4",
                "trade_date": "2026-04-20",
                "trade_time": "12:00:00",
                "contract": "M2S",
                "side_lp": "BUY",
                "quantity_lots": 1.0,
                "price_points": 190.0,
                "counterparty": "Y",
                "counterparty_type": "BROKER",
                "status": "CONFIRMED",
                "is_official_for_calc": True,
                "is_confirmed": True,
                "is_valid_for_calc": True,
                "chrono_key": 4,
            },
        ]
    )

    settings = {
        "commission_bvc_rt": 1.0,
        "commission_broker_rt": 2.0,
        "commission_sgmat_rt": 3.0,
    }
    contract_metrics = compute_contract_metrics(contracts, transactions, settings)
    confirmed_positions = compute_confirmed_positions(transactions, contract_metrics)
    cmp_detail, cmp_summary = compute_cmp_sequential(transactions, contract_metrics)
    cmp_portfolio = build_cmp_portfolio_view(contract_metrics, cmp_summary)

    return {
        "settings": settings,
        "contracts_raw": contracts,
        "transactions_raw": transactions,
        "contracts_validated": contracts,
        "contracts_ready": contracts,
        "contract_issues": pd.DataFrame(),
        "transactions_validated": transactions,
        "transaction_issues": pd.DataFrame(),
        "contract_metrics": contract_metrics,
        "confirmed_positions": confirmed_positions,
        "cmp_detail": cmp_detail,
        "cmp_summary": cmp_summary,
        "cmp_portfolio": cmp_portfolio,
        "global_metrics": compute_global_metrics(contract_metrics),
        "cmp_global_metrics": compute_cmp_global_metrics(cmp_portfolio),
        "alerts": [],
    }


class AnalyticsTests(unittest.TestCase):
    def test_global_metrics_split_open_notional_by_future_side(self):
        state = build_sample_state()
        contract_metrics = state["contract_metrics"]
        global_metrics = state["global_metrics"]

        self.assertEqual(float(global_metrics["total_notional"]), 5000.0)
        self.assertEqual(float(global_metrics["open_notional_futures_long"]), 2000.0)
        self.assertEqual(float(global_metrics["open_notional_futures_short"]), 3000.0)

        by_contract = contract_metrics.set_index("contract_code")
        self.assertEqual(by_contract.loc["M2L", "side_label"], "LONG")
        self.assertEqual(by_contract.loc["M2L", "replication_side_label"], "SHORT")
        self.assertEqual(float(by_contract.loc["M2L", "notional_mad"]), 2000.0)
        self.assertEqual(by_contract.loc["M2S", "side_label"], "SHORT")
        self.assertEqual(by_contract.loc["M2S", "replication_side_label"], "LONG")
        self.assertEqual(float(by_contract.loc["M2S", "notional_mad"]), 3000.0)


class PageSmokeTests(unittest.TestCase):
    def _page_patches(self, state: dict):
        column_config = SimpleNamespace(
            TextColumn=lambda *args, **kwargs: {},
            NumberColumn=lambda *args, **kwargs: {},
            SelectboxColumn=lambda *args, **kwargs: {},
        )

        return [
            patch.object(ui, "init_page", lambda *args, **kwargs: None),
            patch.object(ui, "load_app_state", lambda *args, **kwargs: state),
            patch.object(ui, "render_sidebar_tools", lambda *args, **kwargs: None),
            patch.object(ui, "render_hero", lambda *args, **kwargs: None),
            patch.object(ui, "render_metric_cards", lambda *args, **kwargs: None),
            patch.object(ui, "render_section_header", lambda *args, **kwargs: None),
            patch.object(ui, "render_status_box", lambda *args, **kwargs: None),
            patch.object(ui, "render_footer", lambda *args, **kwargs: None),
            patch.object(ui, "render_data_table", lambda *args, **kwargs: None),
            patch.object(ui, "show_issues", lambda *args, **kwargs: None),
            patch.object(ui, "render_form_group", lambda *args, **kwargs: None),
            patch.object(ui, "render_micro_note", lambda *args, **kwargs: None),
            patch.object(st, "tabs", lambda labels: [DummyContext() for _ in labels]),
            patch.object(st, "columns", lambda spec: [DummyContext() for _ in range(spec if isinstance(spec, int) else len(spec))]),
            patch.object(st, "form", lambda *args, **kwargs: DummyContext()),
            patch.object(st, "subheader", lambda *args, **kwargs: None),
            patch.object(st, "metric", lambda *args, **kwargs: None),
            patch.object(st, "selectbox", lambda label, options, **kwargs: options[0] if options else None),
            patch.object(st, "multiselect", lambda *args, **kwargs: []),
            patch.object(st, "markdown", lambda *args, **kwargs: None),
            patch.object(st, "button", lambda *args, **kwargs: False),
            patch.object(st, "dataframe", lambda *args, **kwargs: None),
            patch.object(st, "data_editor", lambda data, **kwargs: data),
            patch.object(st, "date_input", lambda *args, **kwargs: pd.Timestamp("2026-04-20").date()),
            patch.object(st, "time_input", lambda *args, **kwargs: pd.Timestamp("2026-04-20 09:00:00").time()),
            patch.object(st, "text_input", lambda *args, **kwargs: ""),
            patch.object(st, "number_input", lambda *args, **kwargs: 0.0),
            patch.object(st, "file_uploader", lambda *args, **kwargs: None),
            patch.object(st, "form_submit_button", lambda *args, **kwargs: False),
            patch.object(st, "download_button", lambda *args, **kwargs: None),
            patch.object(st, "warning", lambda *args, **kwargs: None),
            patch.object(st, "info", lambda *args, **kwargs: None),
            patch.object(st, "success", lambda *args, **kwargs: None),
            patch.object(st, "error", lambda *args, **kwargs: None),
            patch.object(st, "caption", lambda *args, **kwargs: None),
            patch.object(st, "write", lambda *args, **kwargs: None),
            patch.object(st, "stop", lambda *args, **kwargs: None),
            patch.object(st, "rerun", lambda *args, **kwargs: None),
            patch.object(st, "column_config", column_config),
        ]

    def test_pnl_pages_smoke(self):
        state = build_sample_state()
        page_paths = [
            PNL_APP_ROOT / "pages" / "dashboard_page.py",
            PNL_APP_ROOT / "pages" / "pnl_global_page.py",
            PNL_APP_ROOT / "pages" / "position_par_contrat_page.py",
            PNL_APP_ROOT / "pages" / "cmp_sequentiel_page.py",
            PNL_APP_ROOT / "pages" / "transactions_page.py",
        ]

        with ExitStack() as stack:
            for page_patch in self._page_patches(state):
                stack.enter_context(page_patch)
            for page_path in page_paths:
                runpy.run_path(str(page_path), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
