from copy import deepcopy
from datetime import date

APP_TITLE = "MASI20 Futures PnL Tracker"
NEAR_EXPIRY_DAYS = 10
CMP_TOLERANCE = 1e-6
FIXED_UNDERLYING_NAME = "MASI20"

CONTRACT_COLUMNS = [
    "contract_code",
    "underlying_name",
    "expiry_date",
    "tick_value",
    "initial_margin_per_lot",
    "settlement_price_points",
    "is_active_lp",
    "position_limit_per_contract",
    "comments",
]

TRANSACTION_COLUMNS = [
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

DAILY_PRICE_COLUMNS = [
    "date",
    "contract_code",
    "settlement_price_points",
]

DEFAULT_SETTINGS = {
    "spot_index": None,
    "risk_free_rate": None,
    "dividend_yield": None,
    "valuation_date": "",
    "default_tick_value": None,
    "commission_bvc_rt": None,
    "commission_broker_rt": None,
    "commission_sgmat_rt": None,
    "default_position_limit_per_contract": None,
}

TRANSACTION_STATUS_OPTIONS = [
    "CONFIRME",
    "ATTENTE",
    "REJETE",
]

STATUS_ALIASES = {
    "CONFIRME": "CONFIRME",
    "CONFIRMED": "CONFIRME",
    "CONFIRMED_OK": "CONFIRME",
    "DONE": "CONFIRME",
    "FILLED": "CONFIRME",
    "PARTIAL": "CONFIRME",
    "PARTIEL": "CONFIRME",
    "ATTENTE": "ATTENTE",
    "PENDING": "ATTENTE",
    "EN ATTENTE": "ATTENTE",
    "REJETE": "REJETE",
    "REJECTED": "REJETE",
    "CANCELLED": "REJETE",
    "CANCELED": "REJETE",
    "ANNULE": "REJETE",
    "ANNULEE": "REJETE",
}

CONFIRMED_STATUSES = {"CONFIRME"}
OFFICIAL_PNL_STATUSES = {"CONFIRME"}


def current_valuation_date_str() -> str:
    return date.today().isoformat()


def default_settings() -> dict:
    settings = deepcopy(DEFAULT_SETTINGS)
    settings["valuation_date"] = current_valuation_date_str()
    return settings
