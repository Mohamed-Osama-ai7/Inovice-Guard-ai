import streamlit as st

def inject_css(theme: str = "dark") -> None:
    """
    Injects the enterprise design system stylesheet supporting both Dark and Light themes.
    Uses centralized semantic tokens and responsive styling across Desktop, Tablet, and Mobile.
    """
    if not theme or theme not in ("dark", "light"):
        theme = st.session_state.get("theme", "dark")

    # Semantic Design Tokens for Dark and Light Themes
    if theme == "light":
        tokens = """
  --bg:               #f8fafc;       /* Clean financial light background */
  --bg2:              #ffffff;       /* Sidebar pure white surface */
  --surface:          #ffffff;       /* Card background */
  --surface-card:     #ffffff;       /* Dedicated KPI surface */
  --surface-hover:    #f1f5f9;       /* Interactive element hover */
  --surface-elevated: #f8fafc;
  
  --border:           #e2e8f0;       /* Subtle, crisp borders */
  --border-light:     #cbd5e1;
  --border-focus:     #2563eb;       /* Focused input border */

  --primary:          #2563eb;       /* Brand Primary Blue */
  --primary-h:        #1d4ed8;       /* Primary Hover */
  --primary-s:        rgba(37, 99, 235, 0.08); /* Soft primary tint */

  --success:          #059669;       /* Emerald Green */
  --success-s:        rgba(5, 150, 105, 0.08);
  --warning:          #d97706;       /* Amber Warning */
  --warning-s:        rgba(217, 119, 6, 0.08);
  --danger:           #dc2626;       /* Crimson Danger */
  --danger-s:         rgba(220, 38, 38, 0.08);
  --critical:         #991b1b;       /* Deep Critical Red */
  --critical-s:       rgba(153, 27, 27, 0.10);

  --text-main:        #0f172a;       /* Slate-900 high contrast */
  --text-light:       #334155;       /* Slate-700 secondary */
  --text-muted:       #64748b;       /* Slate-500 captions & labels */

  --shadow-sm:        0 1px 2px 0 rgba(0, 0, 0, 0.04);
  --shadow-md:        0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  --shadow-lg:        0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
        """
    else:  # dark theme (default)
        tokens = """
  --bg:               #0b1120;       /* Deep SaaS navy background */
  --bg2:              #0f172a;       /* Sidebar deep slate */
  --surface:          #111827;       /* Card background */
  --surface-card:     #162032;       /* Dedicated KPI surface */
  --surface-hover:    #1e293b;       /* Interactive element hover */
  --surface-elevated: #1e293b;
  
  --border:           #263449;       /* Deep crisp slate borders */
  --border-light:     #334155;
  --border-focus:     #3b82f6;       /* Focused input border */

  --primary:          #3b82f6;       /* Electric Blue */
  --primary-h:        #2563eb;       /* Primary Hover */
  --primary-s:        rgba(59, 130, 246, 0.15); /* Soft primary tint */

  --success:          #10b981;       /* Mint Green */
  --success-s:        rgba(16, 185, 129, 0.15);
  --warning:          #f59e0b;       /* Amber Warning */
  --warning-s:        rgba(245, 158, 11, 0.15);
  --danger:           #ef4444;       /* Rose Red */
  --danger-s:         rgba(239, 68, 68, 0.15);
  --critical:         #dc2626;       /* Deep Red */
  --critical-s:       rgba(220, 38, 38, 0.20);

  --text-main:        #f8fafc;       /* Crisp light text */
  --text-light:       #cbd5e1;       /* Slate-300 secondary */
  --text-muted:       #94a3b8;       /* Slate-400 captions & labels */

  --shadow-sm:        0 1px 2px 0 rgba(0, 0, 0, 0.35);
  --shadow-md:        0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
  --shadow-lg:        0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.35);
        """

    st.markdown(
        f"""
<style>
/* ════════════════════════════════════════════════════
   INVOICEGUARD AI  —  Enterprise Design System
   Modern AI Fintech / Receivables Intelligence Shell
   ════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600&display=swap');

:root {{
{tokens}
  --r-sm:          4px;
  --r-md:          8px;
  --r-lg:          12px;

  --font-sans:     'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono:     'Roboto Mono', SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}

/* ── Global Reset & Anti-Overflow ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .stApp {{
  background: var(--bg) !important;
  color: var(--text-main) !important;
  font-family: var(--font-sans) !important;
  max-width: 100vw !important;
  overflow-x: hidden !important;
  box-sizing: border-box !important;
}}

*, *:before, *:after {{
  box-sizing: border-box !important;
}}

/* Clean default chrome while preserving mobile navigation trigger */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ 
  display: none !important; 
}}
header[data-testid="stHeader"] {{
  background: transparent !important;
  pointer-events: none !important;
  height: 0 !important;
  min-height: 0 !important;
}}
header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {{
  pointer-events: auto !important;
  display: flex !important;
  visibility: visible !important;
  position: fixed !important;
  top: 0.65rem !important;
  left: 0.65rem !important;
  z-index: 999999 !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  padding: 0.25rem !important;
  color: var(--text-main) !important;
  box-shadow: var(--shadow-md) !important;
}}

/* ── Container Layout ── */
.main .block-container {{
  max-width: 1380px;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 1.25rem 2.5rem 4rem !important;
}}

@media (max-width: 1199px) {{
  .main .block-container {{
    padding: 1.25rem 1.75rem 3rem !important;
  }}
}}

@media (max-width: 767px) {{
  .main .block-container {{
    padding: 1rem 0.75rem 2.5rem !important;
  }}
}}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {{
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  color: var(--text-main) !important;
  letter-spacing: -0.015em;
  margin-bottom: 0.5rem !important;
}}

h1 {{ font-size: clamp(1.35rem, 3.5vw, 1.75rem) !important; margin-top: 0 !important; }}
h2 {{ font-size: clamp(1.1rem, 2.8vw, 1.3rem) !important; margin-top: 1.25rem !important; }}
h3 {{ font-size: 0.95rem !important; color: var(--text-light) !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }}
p, span, label, div {{ font-family: var(--font-sans); }}

/* ── Top Application Shell Header Bar ── */
.ig-shell-bar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.35rem 0.2rem 0.75rem 0.2rem;
  margin-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
  min-height: 42px;
}}
.ig-shell-brand-group {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  flex-wrap: wrap;
}}
.ig-shell-logo {{
  color: var(--primary);
  display: flex;
  align-items: center;
}}
.ig-shell-title {{
  font-weight: 700;
  color: var(--text-main);
  letter-spacing: -0.01em;
}}
.ig-shell-sep {{
  color: var(--border-light);
  font-size: 0.8rem;
  opacity: 0.7;
}}
.ig-shell-section {{
  text-transform: uppercase;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}}
.ig-shell-page {{
  font-weight: 600;
  color: var(--primary);
  font-size: 0.82rem;
}}
.ig-shell-role-container {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  height: 100%;
}}
.ig-shell-role-pill {{
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.35rem 0.65rem;
  border-radius: var(--r-sm);
  background: var(--surface);
  color: var(--text-light);
  border: 1px solid var(--border);
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  line-height: 1;
}}
.ig-shell-status-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--success);
  display: inline-block;
  box-shadow: 0 0 6px var(--success);
}}

/* ── Page Header Component ── */
.ig-page-header {{
  margin-bottom: 1.5rem;
  padding-bottom: 0.85rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  flex-wrap: wrap;
}}
.ig-page-header-title {{
  font-size: clamp(1.35rem, 3.5vw, 1.7rem);
  font-weight: 700;
  margin: 0 0 0.25rem 0;
  color: var(--text-main);
  letter-spacing: -0.02em;
}}
.ig-page-header-subtitle {{
  font-size: clamp(0.82rem, 2.2vw, 0.92rem);
  color: var(--text-muted);
  margin: 0;
}}
.ig-page-header-right {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}

/* ── Enterprise KPI Cards ── */
.ig-card {{
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.2rem;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease, border-color 0.2s ease;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}}
.ig-card:hover {{
  box-shadow: var(--shadow-md);
  border-color: var(--border-light);
  transform: translateY(-2px);
}}
.ig-kpi-card {{
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 112px;
}}
.ig-kpi-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.4rem;
}}
.ig-kpi-label {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-muted);
}}
.ig-kpi-icon {{
  font-size: 0.95rem;
  opacity: 0.8;
}}
.ig-kpi-value {{
  font-size: clamp(1.35rem, 3vw, 1.85rem);
  font-weight: 700;
  color: var(--text-main);
  letter-spacing: -0.025em;
  line-height: 1.15;
  font-family: var(--font-sans);
  word-break: break-word;
}}
.ig-card-trend {{
  margin-top: 0.6rem;
  font-size: 0.75rem;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  width: fit-content;
}}
.ig-trend-up {{
  background: var(--success-s);
  color: var(--success);
  border: 1px solid var(--success);
}}
.ig-trend-down {{
  background: var(--danger-s);
  color: var(--danger);
  border: 1px solid var(--danger);
}}
.ig-trend-neutral {{
  background: var(--primary-s);
  color: var(--primary);
  border: 1px solid var(--primary);
}}

/* ── Semantic Status Badges ── */
.ig-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  border-radius: var(--r-sm);
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  white-space: nowrap;
}}
.ig-badge.status-good, .ig-badge.status-low {{
  background: var(--success-s);
  color: var(--success);
  border: 1px solid var(--success);
}}
.ig-badge.status-warning, .ig-badge.status-medium {{
  background: var(--warning-s);
  color: var(--warning);
  border: 1px solid var(--warning);
}}
.ig-badge.status-danger, .ig-badge.status-high {{
  background: var(--danger-s);
  color: var(--danger);
  border: 1px solid var(--danger);
}}
.ig-badge.status-critical {{
  background: var(--critical-s);
  color: var(--critical);
  border: 1px solid var(--critical);
  font-weight: 700;
}}
.ig-badge.status-info {{
  background: var(--primary-s);
  color: var(--primary);
  border: 1px solid var(--primary);
}}

/* ── Sidebar Navigation ── */
[data-testid="stSidebar"] {{
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{
  color: var(--text-main);
}}
[data-testid="stSidebar"] .stButton > button {{
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.75rem !important;
  margin-bottom: 0.2rem !important;
  border-radius: var(--r-md) !important;
  transition: all 0.16s ease !important;
  width: 100% !important;
  border: 1px solid transparent !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
  background: transparent !important;
  color: var(--text-muted) !important;
  border-color: transparent !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
  background: var(--surface-hover) !important;
  color: var(--text-main) !important;
  border-color: var(--border) !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: var(--primary) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
  border-color: var(--primary) !important;
}}

/* ── Form Inputs & Selectboxes ── */
div[data-baseweb="select"] > div, input, textarea {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
  border-radius: var(--r-md) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.88rem !important;
}}
div[data-baseweb="select"] > div:hover, input:focus, textarea:focus {{
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 2px var(--primary-s) !important;
}}
div[data-baseweb="popover"], div[data-baseweb="menu"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
  box-shadow: var(--shadow-lg) !important;
}}
div[data-baseweb="menu"] li {{
  background: transparent !important;
  color: var(--text-main) !important;
}}
div[data-baseweb="menu"] li:hover {{
  background: var(--surface-hover) !important;
}}

/* ── Standard Buttons ── */
.stButton > button {{
  border-radius: var(--r-md) !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  padding: 0.45rem 0.9rem !important;
  transition: all 0.16s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
.stButton > button[kind="primary"] {{
  background: var(--primary) !important;
  border: 1px solid var(--primary) !important;
  color: #ffffff !important;
  box-shadow: var(--shadow-sm) !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: var(--primary-h) !important;
  border-color: var(--primary-h) !important;
  transform: translateY(-1px);
  box-shadow: var(--shadow-md) !important;
}}
.stButton > button[kind="secondary"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background: var(--surface-hover) !important;
  border-color: var(--border-light) !important;
  transform: translateY(-1px);
}}

/* ── DataFrames & Tables ── */
.stDataFrame, div[data-testid="stTable"], div.stTable, [data-testid="stDataFrameContainer"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  box-shadow: var(--shadow-sm) !important;
  overflow-x: auto !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
  border-bottom: 1px solid var(--border) !important;
  gap: 0.5rem !important;
}}
[data-testid="stTabs"] [role="tab"] {{
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 0.55rem 0.9rem !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.15s ease, border-color 0.15s ease !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
  color: var(--text-main) !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
  color: var(--primary) !important;
  border-bottom-color: var(--primary) !important;
}}

/* ── Expanders ── */
div[data-testid="stExpander"] {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}}
div[data-testid="stExpander"] summary {{
  color: var(--text-main) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
}}

/* ── Customer 360 Header Panel ── */
.ig-c360-header {{
  display: flex;
  align-items: center;
  gap: 1.25rem;
  padding: 1.25rem 1.5rem;
  background: var(--surface-card);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
  width: 100%;
}}
.ig-c360-name {{
  font-size: clamp(1.25rem, 3.5vw, 1.55rem);
  font-weight: 700;
  color: var(--text-main);
  margin: 0;
  letter-spacing: -0.02em;
}}
.ig-domain-badge {{
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.2rem 0.55rem;
  border-radius: 4px;
  background: var(--primary-s);
  color: var(--primary);
  border: 1px solid var(--primary);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}}

/* ── Empty State ── */
.ig-empty-state {{
  text-align: center;
  padding: 3rem 1.5rem;
  color: var(--text-muted);
  background: var(--surface);
  border: 1px dashed var(--border-light);
  border-radius: var(--r-md);
  width: 100%;
}}
.ig-empty-icon {{ font-size: 2.25rem; margin-bottom: 0.75rem; opacity: 0.6; }}
.ig-empty-title {{ font-weight: 600; font-size: 1.05rem; color: var(--text-main); margin-bottom: 0.4rem; }}

/* ── Feature Importance Bars ── */
.ig-feature-row {{
  display: flex;
  align-items: center;
  gap: 1rem;
  width: 100%;
}}
.ig-feature-name {{
  flex: 0 0 160px;
  font-size: 0.85rem;
  color: var(--text-light);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.ig-feature-bar-wrapper {{
  flex: 1;
}}
.ig-feature-bar {{
  height: 8px;
  border-radius: 4px;
}}
.ig-feature-val {{
  flex: 0 0 80px;
  font-size: 0.85rem;
  color: var(--text-main);
}}

/* ── Touch Target Rules on Mobile ── */
@media (max-width: 767px) {{
  button, input, select, textarea, [data-baseweb="select"] {{
    min-height: 44px !important;
    font-size: 0.95rem !important;
  }}
  .stButton > button {{
    min-height: 44px !important;
  }}
  .ig-c360-header {{
    flex-direction: column !important;
    align-items: flex-start !important;
    padding: 1rem !important;
  }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )
