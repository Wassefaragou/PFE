import json
from pathlib import Path

import pandas as pd

from .config import (
    CONTRACT_COLUMNS,
    DAILY_PRICE_COLUMNS,
    TRANSACTION_COLUMNS,
    default_settings,
)
from .contracts import clear_contract_overrides, enrich_contract_reference

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
SETTINGS_PATH = STORAGE_DIR / "settings.json"
CONTRACTS_PATH = STORAGE_DIR / "contracts.csv"
TRANSACTIONS_PATH = STORAGE_DIR / "transactions.csv"
DAILY_PRICES_PATH = STORAGE_DIR / "daily_prices.csv"
CSV_STORAGE_LAYOUT = (
    (CONTRACTS_PATH, CONTRACT_COLUMNS),
    (TRANSACTIONS_PATH, TRANSACTION_COLUMNS),
    (DAILY_PRICES_PATH, DAILY_PRICE_COLUMNS),
)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _ensure_frame_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = dataframe.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output[columns]


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(default_settings(), handle, indent=2)
    for path, columns in CSV_STORAGE_LAYOUT:
        if not path.exists():
            _empty_frame(columns).to_csv(path, index=False)


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_storage()
    if not path.exists() or path.stat().st_size == 0:
        return _empty_frame(columns)
    return _ensure_frame_columns(pd.read_csv(path), columns)


def _save_csv(path: Path, dataframe: pd.DataFrame, columns: list[str]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    output = _ensure_frame_columns(dataframe, columns)
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    output.to_csv(path, index=False)


def _normalize_settings_payload(payload: dict | None) -> dict:
    raw_settings = payload or {}
    normalized = default_settings()
    for key in normalized:
        if key in raw_settings:
            normalized[key] = raw_settings[key]

    return normalized


def load_settings() -> dict:
    ensure_storage()
    if not SETTINGS_PATH.exists():
        return default_settings()
    with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return _normalize_settings_payload(loaded)


def save_settings(settings: dict) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    merged = _normalize_settings_payload(settings)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)


def load_contracts(*, fallback_initial_margin_per_lot: float | None = None) -> pd.DataFrame:
    contracts = clear_contract_overrides(enrich_contract_reference(_load_csv(CONTRACTS_PATH, CONTRACT_COLUMNS)))
    if "initial_margin_per_lot" not in contracts.columns:
        return contracts

    fallback_margin = pd.to_numeric(pd.Series([fallback_initial_margin_per_lot]), errors="coerce").iloc[0]
    if pd.notna(fallback_margin):
        current_margin = pd.to_numeric(contracts["initial_margin_per_lot"], errors="coerce")
        missing_margin = current_margin.isna()
        contracts["initial_margin_per_lot"] = current_margin.fillna(float(fallback_margin))
        if missing_margin.any():
            save_contracts(contracts)
    return contracts


def save_contracts(dataframe: pd.DataFrame) -> None:
    _save_csv(
        CONTRACTS_PATH,
        clear_contract_overrides(enrich_contract_reference(dataframe, force_contract_code=True)),
        CONTRACT_COLUMNS,
    )


def load_transactions() -> pd.DataFrame:
    return _load_csv(TRANSACTIONS_PATH, TRANSACTION_COLUMNS)


def save_transactions(dataframe: pd.DataFrame) -> None:
    _save_csv(TRANSACTIONS_PATH, dataframe, TRANSACTION_COLUMNS)


def load_daily_prices() -> pd.DataFrame:
    return _load_csv(DAILY_PRICES_PATH, DAILY_PRICE_COLUMNS)


def save_daily_prices(dataframe: pd.DataFrame) -> None:
    _save_csv(DAILY_PRICES_PATH, dataframe, DAILY_PRICE_COLUMNS)


def reset_storage() -> None:
    ensure_storage()
    save_settings(default_settings())
    save_contracts(_empty_frame(CONTRACT_COLUMNS))
    save_transactions(_empty_frame(TRANSACTION_COLUMNS))
    save_daily_prices(_empty_frame(DAILY_PRICE_COLUMNS))
