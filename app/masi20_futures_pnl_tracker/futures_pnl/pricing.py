import numpy as np
import pandas as pd

from .config import NEAR_EXPIRY_DAYS, current_valuation_date_str


def compute_theoretical_prices(contracts_df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    contracts = contracts_df.copy()
    if contracts.empty:
        return contracts

    valuation_date = pd.to_datetime(settings.get("valuation_date"), errors="coerce")
    if pd.isna(valuation_date):
        valuation_date = pd.to_datetime(current_valuation_date_str(), errors="coerce")
    spot_index = pd.to_numeric(settings.get("spot_index"), errors="coerce")
    risk_free_rate = pd.to_numeric(settings.get("risk_free_rate"), errors="coerce")
    dividend_yield = pd.to_numeric(settings.get("dividend_yield"), errors="coerce")

    contracts["expiry_date_parsed"] = pd.to_datetime(contracts["expiry_date"], errors="coerce").dt.normalize()
    contracts["days_to_expiry"] = (contracts["expiry_date_parsed"] - valuation_date).dt.days
    contracts["theoretical_price"] = spot_index * np.exp(
        (risk_free_rate - dividend_yield) * contracts["days_to_expiry"].fillna(0) / 360.0
    )
    use_settlement = contracts["is_active_lp"].fillna(False) & contracts["settlement_price_points"].notna()
    contracts["mtm_price"] = np.where(
        use_settlement,
        contracts["settlement_price_points"],
        contracts["theoretical_price"],
    )
    contracts["mtm_source"] = np.where(use_settlement, "settlement", "theoretical")
    contracts["expiry_alert"] = np.select(
        [
            contracts["days_to_expiry"].lt(0),
            contracts["days_to_expiry"].between(0, NEAR_EXPIRY_DAYS),
        ],
        ["Expired", "Near expiry"],
        default="OK",
    )
    return contracts
