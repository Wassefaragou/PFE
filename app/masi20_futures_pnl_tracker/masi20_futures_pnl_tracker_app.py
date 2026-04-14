from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"


def run() -> None:
    dashboard = st.Page(str(PAGES_DIR / "dashboard_page.py"), title="Dashboard")
    cmp_sequentiel = st.Page(
        str(PAGES_DIR / "cmp_sequentiel_page.py"),
        title="CMP sequentiel",
    )
    parametres = st.Page(str(PAGES_DIR / "parametres_page.py"), title="Parametres")
    referentiel = st.Page(
        str(PAGES_DIR / "referentiel_contrats_page.py"),
        title="Referentiel contrats",
    )
    transactions = st.Page(str(PAGES_DIR / "transactions_page.py"), title="Transactions")
    positions = st.Page(
        str(PAGES_DIR / "position_par_contrat_page.py"),
        title="Position par contrat",
    )
    pnl_global = st.Page(str(PAGES_DIR / "pnl_global_page.py"), title="P&L Global")

    navigation = st.navigation(
        [
            dashboard,
            parametres,
            referentiel,
            transactions,
            positions,
            pnl_global,
            cmp_sequentiel,
        ],
        position="sidebar",
    )

    navigation.run()
