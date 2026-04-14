import numpy as np
import pandas as pd

from .config import NEAR_EXPIRY_DAYS, current_valuation_date_str


def compute_theoretical_prices(contracts_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    contracts = contracts_df.copy()
    if contracts.empty:
        return contracts

    valuation_date = pd.to_datetime(current_valuation_date_str(), errors="coerce")

    contracts["expiry_date_parsed"] = pd.to_datetime(contracts["expiry_date"], errors="coerce").dt.normalize()
    contracts["days_to_expiry"] = (contracts["expiry_date_parsed"] - valuation_date).dt.days
    contracts["settlement_price_points"] = pd.to_numeric(contracts["settlement_price_points"], errors="coerce")
    contracts["mtm_price"] = contracts["settlement_price_points"]

    contracts["mtm_source"] = np.select(
        [
            contracts["settlement_price_points"].notna(),
        ],
        ["contract"],
        default="missing",
    )
    contracts["expiry_alert"] = np.select(
        [
            contracts["days_to_expiry"].lt(0),
            contracts["days_to_expiry"].between(0, NEAR_EXPIRY_DAYS),
        ],
        ["Expired", "Near expiry"],
        default="OK",
    )
    return contracts
