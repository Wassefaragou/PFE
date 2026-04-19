from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PAGES_DIR = APP_DIR / "pages"
PAGE_SPECS = (
    ("dashboard_page.py", "Dashboard"),
    ("parametres_page.py", "Parametres"),
    ("referentiel_contrats_page.py", "Referentiel contrats"),
    ("transactions_page.py", "Transactions"),
    ("position_par_contrat_page.py", "Position par contrat"),
    ("pnl_global_page.py", "P&L Global"),
    ("cmp_sequentiel_page.py", "CMP sequentiel"),
)


def build_page(page_filename: str, title: str) -> st.Page:
    return st.Page(str(PAGES_DIR / page_filename), title=title)


def run() -> None:
    navigation = st.navigation(
        [build_page(page_filename, title) for page_filename, title in PAGE_SPECS],
        position="sidebar",
    )

    navigation.run()
