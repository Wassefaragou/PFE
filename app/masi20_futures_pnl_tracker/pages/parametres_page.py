import pandas as pd
import streamlit as st

from futures_pnl.storage import save_settings
from futures_pnl.ui import (
    format_currency,
    format_number,
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


def _display_currency_total(values: list[object]) -> str:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric_values.empty:
        return "-"
    return format_currency(float(numeric_values.sum()))


default_tick_value_setting = _optional_number(settings.get("default_tick_value"))
commission_bvc_rt_value = _optional_number(settings.get("commission_bvc_rt"))
commission_broker_rt_value = _optional_number(settings.get("commission_broker_rt"))
commission_sgmat_rt_value = _optional_number(settings.get("commission_sgmat_rt"))
default_position_limit_value = _optional_number(settings.get("default_position_limit_per_contract"))

render_hero(
    "Parametres globaux",
    "Parametres globaux de tick, de frais et de limites. Les cours de valorisation se renseignent contrat par contrat dans le referentiel.",
)

render_metric_cards(
    [
        {
            "label": "Tick global",
            "value": "-" if default_tick_value_setting is None else format_number(default_tick_value_setting),
            "glow": "blue",
        },
        {
            "label": "Frais AR / lot",
            "value": _display_currency_total(
                [commission_bvc_rt_value, commission_broker_rt_value, commission_sgmat_rt_value]
            ),
            "glow": "purple",
        },
        {
            "label": "Limite globale",
            "value": "-" if default_position_limit_value is None else format_number(default_position_limit_value),
            "glow": "green",
        },
    ],
    columns=3,
)

render_section_header(
    "Modifier les parametres",
    "Configuration globale hors cours de valorisation.",
    step="01",
    label="Configuration",
)

render_micro_note(
    "Cours de valorisation",
    "Les cours de valorisation ne sont plus saisis ici. Ils se renseignent directement dans le referentiel contrats, ligne par ligne.",
    tone="info",
)

with st.form("settings_form"):
    render_form_group("Bloc contrat", "Parametre global applique a tous les contrats.")
    default_tick_value = st.number_input(
        "Tick global",
        value=default_tick_value_setting,
        step=1.0,
        placeholder="Ex: 10.00",
    )

    render_form_group("Bloc frais", "Frais aller-retour utilisés dans le P&L économique.")
    col1, col2, col3 = st.columns(3)
    commission_bvc_rt = col1.number_input(
        "Commission BVC AR",
        value=commission_bvc_rt_value,
        step=0.5,
        placeholder="Ex: 3.00",
    )
    commission_broker_rt = col2.number_input(
        "Commission broker AR",
        value=commission_broker_rt_value,
        step=0.5,
        placeholder="Ex: 5.00",
    )
    commission_sgmat_rt = col3.number_input(
        "Commission SGMAT AR",
        value=commission_sgmat_rt_value,
        step=0.5,
        placeholder="Ex: 2.00",
    )

    render_form_group("Bloc risque", "Limite de position globale appliquee aux contrats.")
    default_position_limit_per_contract = st.number_input(
        "Limite de position globale",
        value=default_position_limit_value,
        step=1.0,
        placeholder="Ex: 10",
    )

    submitted = st.form_submit_button("Enregistrer les parametres", width="stretch")

if submitted:
    save_settings(
        {
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
    "Resume des parametres globaux utilises par le moteur.",
    step="02",
    label="Synthese",
)
settings_snapshot = pd.DataFrame(
    [
        {"parameter": key, "value": settings[key]}
        for key in [
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
