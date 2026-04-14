import re

import pandas as pd

from .config import FIXED_UNDERLYING_NAME

MONTH_CODES = {
    1: "JAN",
    2: "FEV",
    3: "MAR",
    4: "AVR",
    5: "MAI",
    6: "JUI",
    7: "JUL",
    8: "AOU",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

MONTH_CODE_ALIASES = {
    "JAN": "JAN",
    "FEB": "FEV",
    "FEV": "FEV",
    "MAR": "MAR",
    "APR": "AVR",
    "AVR": "AVR",
    "MAY": "MAI",
    "MAI": "MAI",
    "JUN": "JUI",
    "JUI": "JUI",
    "JUL": "JUL",
    "AUG": "AOU",
    "AOU": "AOU",
    "SEP": "SEP",
    "OCT": "OCT",
    "NOV": "NOV",
    "DEC": "DEC",
}

CONTRACT_CODE_PATTERN = re.compile(r"^(FMASI20)([A-Z]{3})(\d{2})$")

GLOBAL_PARAMETER_COLUMNS = [
    "tick_value",
    "position_limit_per_contract",
]


def normalize_contract_code(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().upper()
    if not text:
        return pd.NA

    match = CONTRACT_CODE_PATTERN.match(text)
    if not match:
        return text

    prefix, month_code, year_code = match.groups()
    return f"{prefix}{MONTH_CODE_ALIASES.get(month_code, month_code)}{year_code}"


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
        normalize_contract_code
    )
    generated_codes = expiry_dates.map(
        lambda value: normalize_contract_code(_ticker_from_timestamp(value)) if pd.notna(value) else pd.NA
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
