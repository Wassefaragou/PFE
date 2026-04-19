import sys
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import TypedDict

import streamlit as st
import streamlit.components.v1 as components


class AppConfig(TypedDict):
    eyebrow: str
    title: str
    accent: str
    runner: Callable[[], None]


ROOT_DIR = Path(__file__).resolve().parent
NAVIGATION_STATE_KEY = "selected_app"
FAVICON_URL = "https://www.google.com/s2/favicons?domain=attijariwafabank.com&sz=128"
HOME_APP_TITLE = "Plateforme FUTURES MASI20"
HOME_APP_SUBTITLE = ""
MODULE_SUBDIRS = (
    "masi20_futures_pnl_tracker",
    "masi20_futures_pricer",
    "masi20_index_replication",
)


def register_module_paths() -> None:
    for subdir in MODULE_SUBDIRS:
        module_path = str(ROOT_DIR / subdir)
        if module_path not in sys.path:
            sys.path.append(module_path)


register_module_paths()

from masi20_futures_pnl_tracker_app import run as run_pnl_tracker
from masi20_futures_pricer_app import run as run_futures_pricer
from masi20_index_replication_app import run as run_index_replication


APP_REGISTRY: dict[str, AppConfig] = {
    "pnl_tracker": {
        "eyebrow": "Desk Monitoring",
        "title": "MASI20 Futures PnL Tracker",
        "accent": "#f59e0b",
        "runner": run_pnl_tracker,
    },
    "futures_pricer": {
        "eyebrow": "Pricing Engine",
        "title": "MASI20 Futures Pricer",
        "accent": "#4facfe",
        "runner": run_futures_pricer,
    },
    "index_replication": {
        "eyebrow": "Portfolio Lab",
        "title": "MASI20 Index Replication",
        "accent": "#00d4aa",
        "runner": run_index_replication,
    },
}


def open_app(app_key: str) -> None:
    st.session_state[NAVIGATION_STATE_KEY] = app_key
    st.rerun()


def apply_home_favicon() -> None:
    components.html(
        f"""
        <script>
        const doc = window.parent.document;
        let link = doc.querySelector("link[rel='icon']") || doc.querySelector("link[rel='shortcut icon']");
        if (!link) {{
            link = doc.createElement("link");
            link.rel = "icon";
            doc.head.appendChild(link);
        }}
        link.href = "{FAVICON_URL}";
        doc.title = "{HOME_APP_TITLE}";
        </script>
        """,
        height=0,
        width=0,
    )


def inject_home_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg-dark: #0b0f19;
            --bg-mid: #111827;
            --bg-card: rgba(255, 255, 255, 0.04);
            --bg-card-strong: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #e8eefc;
            --text-muted: #91a0bd;
            --gold: #f59e0b;
            --purple: #6c63ff;
            --green: #00d4aa;
            --blue: #4facfe;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.08), transparent 30%),
                radial-gradient(circle at top right, rgba(108, 99, 255, 0.10), transparent 35%),
                linear-gradient(160deg, #0b0f19 0%, #111827 45%, #0f172a 100%);
            color: var(--text-primary);
        }

        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif !important;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        button[kind="headerNoPadding"] {
            display: none !important;
        }

        #MainMenu, header[data-testid="stHeader"], footer {
            visibility: hidden;
        }

        .block-container {
            padding-top: 2.35rem;
            padding-bottom: 2.5rem;
            max-width: 1120px;
        }

        .home-section-label {
            margin: 0 0 1.15rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #f8fafc;
            opacity: 0.95;
        }

        .home-hero {
            margin: 0 0 1.6rem;
        }

        .home-title {
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 3.35rem);
            font-weight: 800;
            line-height: 1.02;
            letter-spacing: -0.05em;
            color: var(--text-primary);
        }

        .home-subtitle {
            margin-top: 0.7rem;
            max-width: 640px;
            font-size: 1rem;
            line-height: 1.7;
            color: var(--text-muted);
        }

        .app-card {
            position: relative;
            min-height: 154px;
            padding: 1.15rem 1.1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid var(--glass-border);
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.028) 100%);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(14px);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .app-card::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 5px;
            border-radius: 24px 24px 0 0;
            background: var(--card-accent);
            box-shadow: 0 0 20px color-mix(in srgb, var(--card-accent) 45%, transparent);
        }

        .app-card::after {
            content: "";
            position: absolute;
            top: -18%;
            right: -8%;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: radial-gradient(circle, color-mix(in srgb, var(--card-accent) 22%, transparent) 0%, transparent 70%);
            pointer-events: none;
            opacity: 0.85;
        }

        .app-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 50px rgba(0, 0, 0, 0.28);
            border-color: color-mix(in srgb, var(--card-accent) 28%, rgba(255, 255, 255, 0.10));
        }

        .app-eyebrow {
            display: inline-flex;
            padding: 0.34rem 0.68rem;
            border-radius: 999px;
            background: color-mix(in srgb, var(--card-accent) 16%, rgba(255, 255, 255, 0.06));
            color: var(--card-accent);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            border: 1px solid color-mix(in srgb, var(--card-accent) 28%, rgba(255, 255, 255, 0.06));
        }

        .app-card h3 {
            margin: 1rem 0 0;
            max-width: 280px;
            font-size: 1.18rem;
            line-height: 1.15;
            color: var(--text-primary);
            letter-spacing: -0.03em;
        }

        .stButton > button {
            min-height: 2.85rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: linear-gradient(180deg, #111827 0%, #0b0f19 100%);
            color: #f8fafc;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.22);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .stButton > button:hover {
            border-color: rgba(245, 158, 11, 0.35);
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.28);
        }

        @media (max-width: 900px) {
            .app-card {
                min-height: auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_card(app_config: AppConfig) -> None:
    st.markdown(
        f"""
        <div class="app-card" style="--card-accent: {app_config['accent']};">
            <div class="app-eyebrow">{escape(app_config['eyebrow'])}</div>
            <h3>{escape(app_config['title'])}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    apply_home_favicon()
    inject_home_styles()
    subtitle_html = (
        f'<div class="home-subtitle">{escape(HOME_APP_SUBTITLE)}</div>'
        if HOME_APP_SUBTITLE
        else ""
    )

    st.markdown(
        f"""
        <div class="home-hero">
            <div class="home-title">{escape(HOME_APP_TITLE)}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="home-section-label">Choisir une application</div>', unsafe_allow_html=True)

    app_entries = list(APP_REGISTRY.items())
    columns = st.columns(len(app_entries))

    for column, (app_key, app_config) in zip(columns, app_entries):
        with column:
            render_app_card(app_config)
            if st.button("Ouvrir", key=f"open_{app_key}", use_container_width=True):
                open_app(app_key)


def main() -> None:
    st.session_state.setdefault(NAVIGATION_STATE_KEY, None)

    selected_app = st.session_state[NAVIGATION_STATE_KEY]
    if selected_app in APP_REGISTRY:
        APP_REGISTRY[selected_app]["runner"]()
        return

    render_home()

if __name__ == "__main__":
    main()
