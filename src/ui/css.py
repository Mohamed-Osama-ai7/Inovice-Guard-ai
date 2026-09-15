import streamlit as st

def inject_css() -> None:
    st.markdown(
        """
<style>
/* ════════════════════════════════════════════════════
   INVOICEGUARD AI  —  Enterprise Design System
   Palette: Enterprise Navy, Crisp White, Professional Accents
   Fully Responsive for Desktop, Laptop, Tablet, & Mobile
   ════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
  --bg:            #0b0f19;       /* Deep SaaS background */
  --bg2:           #111827;       /* Slightly lighter for sidebars */
  --surface:       #1f2937;       /* Card backgrounds */
  --surface-hover: #374151;       /* Interactive elements */
  
  --border:        #374151;       /* Crisp borders */
  --border-light:  #4b5563;

  --primary:       #2563eb;       /* Brand Blue */
  --primary-h:     #1d4ed8;       /* Brand Hover */
  --primary-s:     rgba(37,99,235,0.15); /* Soft primary for tags */

  --success:       #10b981;       /* Green */
  --success-s:     rgba(16,185,129,0.15);
  --warning:       #f59e0b;       /* Amber */
  --warning-s:     rgba(245,158,11,0.15);
  --danger:        #ef4444;       /* Red */
  --danger-s:      rgba(239,68,68,0.15);
  --critical:      #b91c1c;       /* Deep Red */
  --critical-s:    rgba(185,28,28,0.15);

  --text-main:     #f9fafb;       /* Pure white text */
  --text-muted:    #9ca3af;       /* Subtitles */
  --text-light:    #d1d5db;

  --r-sm:          4px;
  --r-md:          8px;
  --r-lg:          12px;

  --font-sans:     'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono:     'Roboto Mono', monospace;
  
  --shadow-sm:     0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md:     0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg:     0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* ── Reset & Global Anti-Overflow ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--text-main) !important;
  font-family: var(--font-sans) !important;
  max-width: 100vw !important;
  overflow-x: hidden !important;
  box-sizing: border-box !important;
}

*, *:before, *:after {
  box-sizing: border-box !important;
}

/* Hide Streamlit default chrome except the mobile sidebar toggle */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { 
  display: none !important; 
}
header[data-testid="stHeader"] {
  background: transparent !important;
  pointer-events: none !important;
  height: 0 !important;
  min-height: 0 !important;
}
header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
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
}


/* ── Responsive Container Padding Across Viewports ── */
.main .block-container {
  max-width: 1400px;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 2rem 3rem 4rem !important;
}

@media (max-width: 1199px) {
  .main .block-container {
    padding: 1.5rem 2rem 3rem !important;
  }
}

@media (max-width: 991px) {
  .main .block-container {
    padding: 1.25rem 1.5rem 3rem !important;
  }
}

@media (max-width: 767px) {
  .main .block-container {
    padding: 1rem 0.75rem 2.5rem !important;
  }
}

@media (max-width: 479px) {
  .main .block-container {
    padding: 0.75rem 0.5rem 2rem !important;
  }
}

/* ── Typography Hierarchy ── */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  color: var(--text-main) !important;
  margin-bottom: 0.5rem !important;
  letter-spacing: -0.01em;
}

h1 { font-size: clamp(1.35rem, 4vw, 1.75rem) !important; margin-top: 0 !important; margin-bottom: 1.25rem !important; }
h2 { font-size: clamp(1.1rem, 3vw, 1.25rem) !important; padding-bottom: 0.25rem; margin-top: 1.5rem !important; }
h3 { font-size: clamp(0.9rem, 2.5vw, 1rem) !important; color: var(--text-light) !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem !important;}

/* ── Page Header Component ── */
.ig-page-header {
  margin-bottom: 1.75rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  flex-wrap: wrap;
}
.ig-page-header-title {
  font-size: clamp(1.35rem, 4vw, 1.75rem);
  font-weight: 600;
  margin: 0 0 0.25rem 0;
  color: var(--text-main);
  letter-spacing: -0.02em;
}
.ig-page-header-subtitle {
  font-size: clamp(0.85rem, 2.5vw, 0.95rem);
  color: var(--text-muted);
  margin: 0;
}
.ig-page-header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.ig-env-badge {
  background: var(--surface-hover);
  border: 1px solid var(--border-light);
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-light);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

@media (max-width: 767px) {
  .ig-page-header {
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 0.75rem !important;
  }
  .ig-page-header-right {
    width: 100% !important;
    justify-content: flex-start !important;
  }
}

/* ── Responsive Streamlit Columns ── */
@media (max-width: 767px) {
  div[data-testid="stHorizontalBlock"], div.row-widget.stHorizontal {
    flex-direction: column !important;
    gap: 1rem !important;
  }
  div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
    margin-bottom: 0.5rem !important;
  }
}

/* ── Professional Cards ── */
.ig-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.25rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s, border-color 0.2s;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}
.ig-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-light);
}
.ig-card-title {
  font-size: clamp(0.75rem, 2.5vw, 0.85rem);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.ig-card-value {
  font-size: clamp(1.3rem, 4vw, 1.875rem);
  font-weight: 700;
  color: var(--text-main);
  letter-spacing: -0.02em;
  line-height: 1.2;
  word-break: break-word;
}
.ig-card-trend {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 0.25rem;
  flex-wrap: wrap;
}
.ig-trend-up { color: var(--success); }
.ig-trend-down { color: var(--danger); }
.ig-trend-neutral { color: var(--text-light); }

@media (max-width: 767px) {
  .ig-card {
    padding: 1rem !important;
  }
}

/* ── Semantic Badges ── */
.ig-badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: var(--r-sm);
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.ig-badge.status-low, .ig-badge.status-good { background: var(--success-s); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.ig-badge.status-medium, .ig-badge.status-warning { background: var(--warning-s); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }
.ig-badge.status-high, .ig-badge.status-danger { background: var(--danger-s); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
.ig-badge.status-critical { background: var(--critical-s); color: #fca5a5; border: 1px solid rgba(239,68,68,0.5); }
.ig-badge.status-info { background: var(--primary-s); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }

/* ── Sidebar (App Shell & Navigation) ── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1 {
  font-size: 1.25rem !important;
  color: var(--text-main) !important;
  margin-bottom: 0.25rem !important;
}
[data-testid="stSidebar"] .stMarkdown p {
  font-size: 0.8rem !important;
  color: var(--text-muted) !important;
  margin-bottom: 1rem !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  padding: 0.6rem 0.75rem !important;
  border-radius: var(--r-sm);
  margin-bottom: 0.1rem;
  transition: background 0.15s, color 0.15s;
  min-height: 40px !important;
  display: flex !important;
  align-items: center !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  background: var(--surface-hover);
}
[data-testid="stSidebar"] div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  color: var(--text-light) !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div {
  background: var(--primary-s);
  border-radius: var(--r-sm);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div p {
  color: #60a5fa !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] div:first-child {
  display: none !important;
}

/* ── Sidebar Navigation Buttons ── */
[data-testid="stSidebar"] .stButton > button {
  text-align: left !important;
  justify-content: flex-start !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.75rem !important;
  margin-bottom: 0.2rem !important;
  border-radius: var(--r-md) !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
  border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-color: transparent !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background: var(--surface) !important;
  color: var(--text-main) !important;
  border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--primary) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
  border-color: var(--primary) !important;
}


/* ── Custom Responsive Data Tables ── */
.stDataFrame, div[data-testid="stTable"], div.stTable, [data-testid="stDataFrameContainer"] {
  width: 100% !important;
  max-width: 100% !important;
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch !important;
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
}
.stDataFrame [data-testid="StyledFullScreenButton"] {
  display: none !important;
}

/* ── Responsive Plotly Charts ── */
.js-plotly-plot, .plot-container, .plotly {
  width: 100% !important;
  max-width: 100% !important;
  overflow-x: hidden !important;
}

/* ── Customer 360 Header Panel ── */
.ig-c360-header {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  padding: 1.25rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  margin-bottom: 1.25rem;
  width: 100%;
}
.ig-c360-avatar {
  width: 56px;
  height: 56px;
  background: var(--surface-hover);
  border-radius: var(--r-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.75rem;
  border: 1px solid var(--border-light);
  flex-shrink: 0;
}
.ig-c360-info { flex: 1; min-width: 0; }
.ig-c360-name {
  font-size: clamp(1.2rem, 3.5vw, 1.5rem);
  font-weight: 700;
  color: var(--text-main);
  margin: 0 0 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.ig-c360-meta {
  display: flex;
  gap: 1.25rem;
  color: var(--text-muted);
  font-size: 0.88rem;
  flex-wrap: wrap;
}
.ig-c360-meta span {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

@media (max-width: 767px) {
  .ig-c360-header {
    flex-direction: column !important;
    align-items: flex-start !important;
    padding: 1rem !important;
    gap: 1rem !important;
  }
  .ig-c360-meta {
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 0.5rem !important;
  }
}

/* ── Responsive Feature Importance Bars ── */
.ig-feature-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  width: 100%;
}
.ig-feature-name {
  flex: 0 0 160px;
  font-size: 0.85rem;
  color: var(--text-light);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ig-feature-bar-wrapper {
  flex: 1;
}
.ig-feature-bar {
  height: 8px;
  border-radius: 4px;
}
.ig-feature-val {
  flex: 0 0 80px;
  font-size: 0.85rem;
  color: var(--text-main);
}

@media (max-width: 767px) {
  .ig-feature-row {
    flex-wrap: wrap;
    gap: 0.25rem 0.75rem;
    margin-bottom: 0.5rem;
  }
  .ig-feature-name {
    flex: 1 1 auto;
    text-align: left;
    font-size: 0.8rem;
  }
  .ig-feature-val {
    flex: 0 0 auto;
    font-size: 0.8rem;
  }
  .ig-feature-bar-wrapper {
    flex: 1 1 100%;
    margin-top: 0.2rem;
  }
}

/* ── Touch UX & Form Controls ── */
@media (max-width: 767px) {
  button, input, select, textarea, [data-baseweb="select"], div[role="button"] {
    min-height: 44px !important;
    font-size: 0.95rem !important;
  }
  .stButton > button {
    width: 100% !important;
    min-height: 44px !important;
  }
}

/* ── Responsive Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  overflow-x: auto !important;
  white-space: nowrap !important;
  -webkit-overflow-scrolling: touch !important;
  max-width: 100% !important;
  padding-bottom: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
  padding: 0.5rem 0.75rem !important;
  font-size: 0.85rem !important;
}

/* ── Recommendation Callouts ── */
.ig-recommendation {
  background: var(--surface);
  border-left: 4px solid var(--primary);
  padding: 1.15rem;
  border-radius: 0 var(--r-md) var(--r-md) 0;
  margin-bottom: 1rem;
  width: 100%;
  box-sizing: border-box;
}
.ig-recommendation-title {
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 0.4rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.ig-recommendation-body {
  color: var(--text-light);
  font-size: 0.92rem;
  line-height: 1.5;
}

/* ── Insight Signals ── */
.ig-signal {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
  width: 100%;
}
.ig-signal:last-child { border-bottom: none; }
.ig-signal-icon { font-size: 1.1rem; margin-top: 0.1rem; flex-shrink: 0; }
.ig-signal-content { flex: 1; min-width: 0; }
.ig-signal-title { font-weight: 500; color: var(--text-main); margin-bottom: 0.25rem; }
.ig-signal-desc { font-size: 0.85rem; color: var(--text-muted); }

/* ── Utility Classes ── */
.ig-text-mono { font-family: var(--font-mono); }
.ig-empty-state {
  text-align: center;
  padding: 3rem 1.5rem;
  color: var(--text-muted);
  background: var(--surface);
  border: 1px dashed var(--border-light);
  border-radius: var(--r-md);
  width: 100%;
}
.ig-empty-icon { font-size: 2.25rem; margin-bottom: 0.75rem; opacity: 0.5; }
.ig-empty-title { font-weight: 600; font-size: 1.05rem; color: var(--text-main); margin-bottom: 0.4rem; }
</style>
        """,
        unsafe_allow_html=True,
    )
