import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import DASHBOARD_HISTORY_COLUMNS, current_valuation_date_str
from .storage import load_dashboard_history, save_dashboard_history


DASHBOARD_GLOBAL_METRIC_KEYS = [
    "total_management_pnl",
    "total_accounting_pnl",
    "total_unrealized_pnl",
    "total_realized_pnl",
    "total_commissions",
    "open_notional_futures_long",
    "open_notional_futures_short",
    "global_exposure",
    "total_notional",
]


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _serialize_mapping(payload: dict) -> str:
    return json.dumps(
        {str(key): _json_safe_value(value) for key, value in payload.items()},
        ensure_ascii=True,
    )


def _serialize_records(records: list[dict]) -> str:
    safe_records = []
    for record in records:
        safe_records.append({str(key): _json_safe_value(value) for key, value in record.items()})
    return json.dumps(safe_records, ensure_ascii=True)


def _serialize_dataframe(dataframe: pd.DataFrame) -> str:
    if dataframe is None or dataframe.empty:
        return "[]"
    return _serialize_records(dataframe.to_dict(orient="records"))


def _deserialize_records(payload: object) -> list[dict]:
    if payload is None:
        return []
    try:
        if pd.isna(payload):
            return []
    except (TypeError, ValueError):
        pass

    text = str(payload).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _deserialize_mapping(payload: object) -> dict:
    if payload is None:
        return {}
    try:
        if pd.isna(payload):
            return {}
    except (TypeError, ValueError):
        pass

    text = str(payload).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _deserialize_dataframe(payload: object) -> pd.DataFrame:
    return pd.DataFrame(_deserialize_records(payload))


def _numeric_value(value: object, fallback: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return fallback
    return float(numeric)


def _count_placeholder(count_value: object) -> pd.DataFrame:
    count = int(max(_numeric_value(count_value, fallback=0.0), 0.0))
    return pd.DataFrame(index=range(count))


def _open_cmp_view(portfolio_view: pd.DataFrame) -> pd.DataFrame:
    if portfolio_view.empty or "abs_position" not in portfolio_view.columns:
        return portfolio_view.head(0).copy()
    return portfolio_view.loc[portfolio_view["abs_position"].fillna(0.0) > 0].copy()


def build_dashboard_snapshot(app_state: dict, snapshot_date: str | None = None) -> dict:
    snapshot_date = snapshot_date or current_valuation_date_str()
    global_metrics = app_state["global_metrics"]
    portfolio_view = app_state["contract_metrics"].copy()
    open_cmp_view = _open_cmp_view(portfolio_view)
    confirmed_positions = app_state["confirmed_positions"].copy()
    alerts = app_state["alerts"]

    snapshot = {
        "date": snapshot_date,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "contract_count": len(app_state["contracts_raw"]),
        "transaction_count": len(app_state["transactions_raw"]),
        "open_contract_count": int((portfolio_view["abs_position"] > 0).sum())
        if not portfolio_view.empty and "abs_position" in portfolio_view.columns
        else 0,
        "global_metrics_json": _serialize_mapping(global_metrics),
        "alerts_json": _serialize_records(alerts),
        "portfolio_json": _serialize_dataframe(portfolio_view),
        "open_cmp_json": _serialize_dataframe(open_cmp_view),
        "confirmed_positions_json": _serialize_dataframe(confirmed_positions),
    }

    for key in DASHBOARD_GLOBAL_METRIC_KEYS:
        snapshot[key] = _json_safe_value(global_metrics.get(key, 0.0))

    return snapshot


def _normalize_dashboard_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return pd.Timestamp(parsed).strftime("%Y-%m-%d")
    return str(value).strip()


def merge_dashboard_snapshot(history: pd.DataFrame, snapshot: dict) -> tuple[pd.DataFrame, bool]:
    snapshot = dict(snapshot)
    snapshot["date"] = _normalize_dashboard_date(snapshot.get("date"))
    snapshot_date = snapshot["date"]

    if history.empty:
        updated = pd.DataFrame([snapshot], columns=DASHBOARD_HISTORY_COLUMNS)
        return updated, False

    normalized_history = history.copy()
    normalized_history["date"] = normalized_history["date"].map(_normalize_dashboard_date)

    known_dates = set(normalized_history["date"].dropna().astype(str))
    is_new_day = snapshot_date not in known_dates

    previous_days = normalized_history.loc[normalized_history["date"] != snapshot_date].copy()
    updated = pd.concat(
        [previous_days, pd.DataFrame([snapshot], columns=DASHBOARD_HISTORY_COLUMNS)],
        ignore_index=True,
    )
    updated = updated.sort_values("date").reset_index(drop=True)
    return updated, is_new_day


def upsert_today_dashboard_snapshot(app_state: dict) -> pd.DataFrame:
    snapshot = build_dashboard_snapshot(app_state)
    history = load_dashboard_history()
    updated, is_new_day = merge_dashboard_snapshot(history, snapshot)
    save_dashboard_history(updated)
    saved_history = load_dashboard_history()
    saved_history.attrs["snapshot_date"] = str(snapshot["date"])
    saved_history.attrs["is_new_dashboard_day"] = bool(is_new_day)
    return saved_history


def dashboard_history_dates(history_df: pd.DataFrame) -> list[str]:
    if history_df.empty or "date" not in history_df.columns:
        return []
    return sorted(history_df["date"].dropna().astype(str).unique().tolist(), reverse=True)


def get_dashboard_snapshot(history_df: pd.DataFrame, selected_date: str | None) -> dict | None:
    dates = dashboard_history_dates(history_df)
    if not dates:
        return None

    selected = selected_date if selected_date in dates else dates[0]
    matches = history_df.loc[history_df["date"].astype(str) == selected]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def dashboard_state_from_snapshot(snapshot: dict | None, fallback_state: dict) -> dict:
    if snapshot is None:
        return fallback_state

    state = dict(fallback_state)
    global_metrics = dict(fallback_state.get("global_metrics", {}))
    global_metrics.update(_deserialize_mapping(snapshot.get("global_metrics_json")))
    for key in DASHBOARD_GLOBAL_METRIC_KEYS:
        fallback_metric = _numeric_value(global_metrics.get(key), fallback=0.0)
        global_metrics[key] = _numeric_value(snapshot.get(key), fallback=fallback_metric)
    global_metrics["global_exposure"] = (
        _numeric_value(global_metrics.get("open_notional_futures_long"), fallback=0.0)
        + _numeric_value(global_metrics.get("open_notional_futures_short"), fallback=0.0)
    )

    portfolio_view = _deserialize_dataframe(snapshot.get("portfolio_json"))
    open_cmp_view = _deserialize_dataframe(snapshot.get("open_cmp_json"))
    confirmed_positions = _deserialize_dataframe(snapshot.get("confirmed_positions_json"))

    state["global_metrics"] = global_metrics
    state["contract_metrics"] = portfolio_view
    state["dashboard_portfolio_view"] = portfolio_view
    state["dashboard_open_cmp_view"] = open_cmp_view
    state["confirmed_positions"] = confirmed_positions
    state["alerts"] = _deserialize_records(snapshot.get("alerts_json"))
    state["contracts_raw"] = _count_placeholder(snapshot.get("contract_count"))
    state["transactions_raw"] = _count_placeholder(snapshot.get("transaction_count"))
    state["dashboard_history_date"] = str(snapshot.get("date", ""))

    return state
