import pandas as pd

from .config import FIXED_UNDERLYING_NAME

MONTH_CODES = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

GLOBAL_PARAMETER_COLUMNS = [
    "tick_value",
    "position_limit_per_contract",
]


def _ticker_from_timestamp(expiry_date: pd.Timestamp) -> str:
    return f"F{FIXED_UNDERLYING_NAME}{MONTH_CODES[int(expiry_date.month)]}{expiry_date.strftime('%y')}"


def enrich_contract_reference(
    dataframe: pd.DataFrame | None,
    *,
    force_contract_code: bool = False,
) -> pd.DataFrame:
    contracts = dataframe.copy() if dataframe is not None else pd.DataFrame()
    if "expiry_date" not in contracts.columns:
        contracts["expiry_date"] = pd.NA
    if "contract_code" not in contracts.columns:
        contracts["contract_code"] = pd.NA

    expiry_dates = pd.to_datetime(contracts["expiry_date"], errors="coerce").dt.normalize()
    existing_codes = contracts["contract_code"].map(
        lambda value: str(value).strip().upper() if pd.notna(value) and str(value).strip() else pd.NA
    )
    generated_codes = expiry_dates.map(
        lambda value: _ticker_from_timestamp(value) if pd.notna(value) else pd.NA
    )

    contracts["expiry_date"] = expiry_dates.dt.strftime("%Y-%m-%d")
    contracts.loc[expiry_dates.isna(), "expiry_date"] = pd.NA
    contracts["contract_code"] = generated_codes if force_contract_code else existing_codes.fillna(generated_codes)
    contracts["underlying_name"] = FIXED_UNDERLYING_NAME
    return contracts


def clear_contract_overrides(dataframe: pd.DataFrame | None) -> pd.DataFrame:
    contracts = dataframe.copy() if dataframe is not None else pd.DataFrame()
    for column in GLOBAL_PARAMETER_COLUMNS:
        contracts[column] = pd.NA
    return contracts
