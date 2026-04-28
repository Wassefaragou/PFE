from copy import deepcopy
from datetime import date

APP_TITLE = "MASI20 Futures PnL Tracker"
NEAR_EXPIRY_DAYS = 10
FIXED_UNDERLYING_NAME = "MASI20"

CONTRACT_COLUMNS = [
    "contract_code",
    "underlying_name",
    "expiry_date",
    "tick_value",
    "settlement_price_points",
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

DASHBOARD_HISTORY_COLUMNS = [
    "date",
    "updated_at",
    "total_management_pnl",
    "total_accounting_pnl",
    "total_unrealized_pnl",
    "total_realized_pnl",
    "total_commissions",
    "open_notional_futures_long",
    "open_notional_futures_short",
    "global_exposure",
    "total_notional",
    "contract_count",
    "transaction_count",
    "open_contract_count",
    "global_metrics_json",
    "alerts_json",
    "portfolio_json",
    "open_cmp_json",
    "confirmed_positions_json",
]

DEFAULT_SETTINGS = {
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
    return deepcopy(DEFAULT_SETTINGS)
