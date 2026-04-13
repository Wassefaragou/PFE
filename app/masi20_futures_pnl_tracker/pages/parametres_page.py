import pandas as pd
import streamlit as st

from futures_pnl.storage import save_settings
from futures_pnl.ui import (
    format_currency,
    format_number,
    format_pct,
    init_page,
    load_app_state,
    render_data_table,
    render_footer,
    render_form_group,
    render_hero,
    render_metric_cards,
    render_micro_note,
    render_section_header,
    render_sidebar_tools,
)

init_page("Parametres")
state = load_app_state()
render_sidebar_tools(state)
settings = state["settings"]


def _optional_number(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _display_number(value: object, *, fmt: str = "{:,.2f}") -> str:
    numeric = _optional_number(value)
    return "-" if numeric is None else fmt.format(numeric)


spot_index_value = _optional_number(settings.get("spot_index"))
risk_free_rate_value = _optional_number(settings.get("risk_free_rate"))
dividend_yield_value = _optional_number(settings.get("dividend_yield"))
default_tick_value_setting = _optional_number(settings.get("default_tick_value"))
commission_bvc_rt_value = _optional_number(settings.get("commission_bvc_rt"))
commission_broker_rt_value = _optional_number(settings.get("commission_broker_rt"))
commission_sgmat_rt_value = _optional_number(settings.get("commission_sgmat_rt"))
default_position_limit_value = _optional_number(settings.get("default_position_limit_per_contract"))

render_hero(
    "Parametres globaux",
    "Parametres communs du pricing, des frais et des limites globales.",
    badges=[("Prix", ""), ("Frais", "purple"), ("Limites", "green")],
)

render_metric_cards(
    [
        {"label": "Indice spot", "value": _display_number(spot_index_value), "glow": "gold"},
        {
            "label": "Taux sans risque",
            "value": "-" if risk_free_rate_value is None else format_pct(risk_free_rate_value),
            "glow": "purple",
        },
        {
            "label": "Rendement dividende",
            "value": "-" if dividend_yield_value is None else format_pct(dividend_yield_value),
            "glow": "green",
        },
        {
            "label": "Tick global",
            "value": "-" if default_tick_value_setting is None else format_number(default_tick_value_setting),
            "glow": "blue",
        },
    ],
    columns=4,
)

render_section_header(
    "Modifier les parametres",
    "Ces valeurs servent de reference globale pour le pricing, les frais et les limites.",
    step="01",
    label="Configuration",
)

render_micro_note(
    "Parametres communs",
    "Le tick et la limite restent globaux. La marge initiale est desormais definie dans le referentiel contrat par contrat.",
    tone="info",
)

with st.form("settings_form"):
    render_form_group("Bloc marche", "Parametres utilises pour le prix theorique et la valorisation.")
    col1, col2, col3 = st.columns(3)
    spot_index = col1.number_input("Indice spot", value=spot_index_value, step=1.0, placeholder="Ex: 1500.00")
    risk_free_rate = col2.number_input(
        "Taux sans risque",
        value=risk_free_rate_value,
        step=0.0001,
        format="%.4f",
        placeholder="Ex: 0.0275",
    )
    dividend_yield = col3.number_input(
        "Rendement dividende",
        value=dividend_yield_value,
        step=0.0001,
        format="%.4f",
        placeholder="Ex: 0.0120",
    )

    render_form_group("Bloc contrat", "Parametres globaux appliques a tous les contrats MASI20.")
    col4, col5 = st.columns(2)
    col4.text_input(
        "Date de valorisation",
        value=str(settings["valuation_date"]),
        disabled=True,
        help="La date de valorisation est automatiquement la date du jour de l'application.",
    )
    default_tick_value = col5.number_input(
        "Tick global",
        value=default_tick_value_setting,
        step=1.0,
        placeholder="Ex: 10.00",
    )

    render_form_group("Bloc frais", "Frais aller-retour utilises dans le P&L economique.")
    col7, col8, col9 = st.columns(3)
    commission_bvc_rt = col7.number_input(
        "Commission BVC AR",
        value=commission_bvc_rt_value,
        step=0.5,
        placeholder="Ex: 3.00",
    )
    commission_broker_rt = col8.number_input(
        "Commission broker AR",
        value=commission_broker_rt_value,
        step=0.5,
        placeholder="Ex: 5.00",
    )
    commission_sgmat_rt = col9.number_input(
        "Commission SGMAT AR",
        value=commission_sgmat_rt_value,
        step=0.5,
        placeholder="Ex: 2.00",
    )

    render_form_group("Bloc risque", "Limites de position globales.")
    col10, col11 = st.columns(2)
    default_position_limit_per_contract = col10.number_input(
        "Limite de position globale",
        value=default_position_limit_value,
        step=1.0,
        placeholder="Ex: 10",
    )

    submitted = st.form_submit_button("Enregistrer les parametres", width="stretch")

if submitted:
    save_settings(
        {
            "spot_index": spot_index,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "default_tick_value": default_tick_value,
            "commission_bvc_rt": commission_bvc_rt,
            "commission_broker_rt": commission_broker_rt,
            "commission_sgmat_rt": commission_sgmat_rt,
            "default_position_limit_per_contract": default_position_limit_per_contract,
        }
    )
    st.rerun()

render_section_header(
    "Valeurs actuelles",
    "Resume des parametres actuellement utilises par le moteur.",
    step="02",
    label="Synthese",
)
settings_snapshot = pd.DataFrame(
    [
        {"parameter": key, "value": settings[key]}
        for key in [
            "spot_index",
            "risk_free_rate",
            "dividend_yield",
            "valuation_date",
            "default_tick_value",
            "commission_bvc_rt",
            "commission_broker_rt",
            "commission_sgmat_rt",
            "default_position_limit_per_contract",
        ]
        if key in settings
    ]
)
render_data_table(settings_snapshot)

render_footer()
