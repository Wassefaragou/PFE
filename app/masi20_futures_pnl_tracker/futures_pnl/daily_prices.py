import json

import pandas as pd

from .config import DAILY_PRICE_COLUMNS, current_valuation_date_str
from .contracts import normalize_contract_code


def _empty_daily_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_PRICE_COLUMNS)


def _normalize_date(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize().strftime("%Y-%m-%d")


def _numeric_value(value: object) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _json_records(payload: object) -> list[dict]:
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


def normalize_daily_prices(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    prices = dataframe.copy() if dataframe is not None else _empty_daily_prices()
    for column in DAILY_PRICE_COLUMNS:
        if column not in prices.columns:
            prices[column] = pd.NA

    prices = prices[DAILY_PRICE_COLUMNS].copy()
    prices["date"] = prices["date"].map(_normalize_date)
    prices["contract_code"] = prices["contract_code"].map(normalize_contract_code)
    prices["settlement_price_points"] = pd.to_numeric(
        prices["settlement_price_points"],
        errors="coerce",
    )

    prices = prices.loc[
        prices["date"].notna()
        & prices["contract_code"].notna()
        & prices["settlement_price_points"].notna()
        & prices["settlement_price_points"].gt(0)
    ].copy()

    if prices.empty:
        return _empty_daily_prices()

    return (
        prices.drop_duplicates(["date", "contract_code"], keep="last")
        .sort_values(["date", "contract_code"])
        .reset_index(drop=True)
    )


def _daily_price_records_from_dashboard_history(history_df: pd.DataFrame | None) -> list[dict]:
    if history_df is None or history_df.empty:
        return []

    records: list[dict] = []
    for row in history_df.itertuples(index=False):
        snapshot_date = _normalize_date(getattr(row, "date", None))
        if snapshot_date is None:
            continue

        for position in _json_records(getattr(row, "portfolio_json", None)):
            contract_code = normalize_contract_code(position.get("contract_code"))
            settlement_price = _numeric_value(
                position.get("mtm_price", position.get("settlement_price_points"))
            )
            if pd.isna(contract_code) or settlement_price is None or settlement_price <= 0:
                continue
            records.append(
                {
                    "date": snapshot_date,
                    "contract_code": contract_code,
                    "settlement_price_points": settlement_price,
                }
            )
    return records


def backfill_daily_prices_from_dashboard_history(
    daily_prices_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
) -> pd.DataFrame:
    backfill_records = _daily_price_records_from_dashboard_history(history_df)
    if not backfill_records:
        return normalize_daily_prices(daily_prices_df)

    frames = [pd.DataFrame(backfill_records, columns=DAILY_PRICE_COLUMNS)]
    existing_prices = normalize_daily_prices(daily_prices_df)
    if not existing_prices.empty:
        frames.append(existing_prices)
    return normalize_daily_prices(pd.concat(frames, ignore_index=True))


def current_daily_price_records(
    contracts_df: pd.DataFrame | None,
    valuation_date: object | None = None,
) -> pd.DataFrame:
    if contracts_df is None or contracts_df.empty:
        return _empty_daily_prices()

    price_date = _normalize_date(valuation_date or current_valuation_date_str())
    if price_date is None:
        return _empty_daily_prices()

    contracts = contracts_df.copy()
    contracts["contract_code"] = contracts["contract_code"].map(normalize_contract_code)
    contracts["settlement_price_points"] = pd.to_numeric(
        contracts.get("settlement_price_points"),
        errors="coerce",
    )
    if "is_valid" in contracts.columns:
        contracts = contracts.loc[contracts["is_valid"].fillna(False)].copy()

    records = contracts.loc[
        contracts["contract_code"].notna()
        & contracts["settlement_price_points"].notna()
        & contracts["settlement_price_points"].gt(0),
        ["contract_code", "settlement_price_points"],
    ].copy()
    if records.empty:
        return _empty_daily_prices()

    records.insert(0, "date", price_date)
    return normalize_daily_prices(records)


def upsert_current_daily_prices(
    daily_prices_df: pd.DataFrame | None,
    contracts_df: pd.DataFrame | None,
    valuation_date: object | None = None,
) -> pd.DataFrame:
    prices = normalize_daily_prices(daily_prices_df)
    current_prices = current_daily_price_records(contracts_df, valuation_date=valuation_date)
    if current_prices.empty:
        return prices
    if prices.empty:
        return current_prices

    current_keys = set(zip(current_prices["date"], current_prices["contract_code"]))
    keep_mask = ~prices[["date", "contract_code"]].apply(tuple, axis=1).isin(current_keys)
    return normalize_daily_prices(
        pd.concat([prices.loc[keep_mask], current_prices], ignore_index=True)
    )


def sync_daily_prices(
    daily_prices_df: pd.DataFrame | None,
    contracts_df: pd.DataFrame | None,
    history_df: pd.DataFrame | None,
    valuation_date: object | None = None,
) -> pd.DataFrame:
    backfilled = backfill_daily_prices_from_dashboard_history(daily_prices_df, history_df)
    return upsert_current_daily_prices(backfilled, contracts_df, valuation_date=valuation_date)


def latest_price_date_before(
    daily_prices_df: pd.DataFrame | None,
    valuation_date: object | None = None,
) -> str | None:
    prices = normalize_daily_prices(daily_prices_df)
    if prices.empty:
        return None

    current_date = _normalize_date(valuation_date or current_valuation_date_str())
    if current_date is None:
        return None

    previous_dates = prices.loc[prices["date"] < current_date, "date"].dropna()
    if previous_dates.empty:
        return None
    return str(previous_dates.max())


def price_map_for_date(
    daily_prices_df: pd.DataFrame | None,
    price_date: object | None,
) -> dict[str, float]:
    normalized_date = _normalize_date(price_date)
    if normalized_date is None:
        return {}

    prices = normalize_daily_prices(daily_prices_df)
    if prices.empty:
        return {}

    day_prices = prices.loc[prices["date"] == normalized_date]
    return dict(zip(day_prices["contract_code"], day_prices["settlement_price_points"]))
