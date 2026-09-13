import streamlit as st

def inject_css() -> None:
    st.markdown(
        """
<style>
/* ════════════════════════════════════════════════════
   INVOICEGUARD AI  —  Enterprise Design System
   Palette: Enterprise Navy, Crisp White, Professional Accents
   ════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
  --bg:          #0b0f19;       /* Deep SaaS background */
  --bg2:         #111827;       /* Slightly lighter for sidebars */
  --surface:     #1f2937;       /* Card backgrounds */
  --surface-hover: #374151;     /* Interactive elements */
  
  --border:      #374151;       /* Crisp borders */
  --border-light:#4b5563;

  --primary:     #2563eb;       /* Brand Blue */
  --primary-h:   #1d4ed8;       /* Brand Hover */
  --primary-s:   rgba(37,99,235,0.15); /* Soft primary for tags */

  --success:     #10b981;       /* Green */
  --success-s:   rgba(16,185,129,0.15);
  --warning:     #f59e0b;       /* Amber */
  --warning-s:   rgba(245,158,11,0.15);
  --danger:      #ef4444;       /* Red */
  --danger-s:    rgba(239,68,68,0.15);
  --critical:    #b91c1c;       /* Deep Red */
  --critical-s:  rgba(185,28,28,0.15);

  --text-main:   #f9fafb;       /* Pure white text */
  --text-muted:  #9ca3af;       /* Subtitles */
  --text-light:  #d1d5db;

  --r-sm:        4px;
  --r-md:        8px;
  --r-lg:        12px;

  --font-sans:   'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono:   'Roboto Mono', monospace;
  
  --shadow-sm:   0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md:   0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg:   0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

/* ── Reset & Global ── */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg) !important;
  color: var(--text-main) !important;
  font-family: var(--font-sans) !important;
}

/* Hide Streamlit default UI elements */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

/* ── Layout Spacing ── */
.main .block-container {
  padding: 2rem 3rem 4rem !important;
  max-width: 1400px;
}

/* ── Typography Hierarchy ── */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  color: var(--text-main) !important;
  margin-bottom: 0.5rem !important;
  letter-spacing: -0.01em;
}

h1 { font-size: 1.75rem !important; margin-top: 0 !important; margin-bottom: 1.5rem !important; }
h2 { font-size: 1.25rem !important; padding-bottom: 0.25rem; margin-top: 2rem !important; }
h3 { font-size: 1rem !important; color: var(--text-light) !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem !important;}

/* ── Page Header Component ── */
.ig-page-header {
  margin-bottom: 2rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
}
.ig-page-header-left {}
.ig-page-header-title {
  font-size: 1.75rem;
  font-weight: 600;
  margin: 0 0 0.5rem 0;
  color: var(--text-main);
  letter-spacing: -0.02em;
}
.ig-page-header-subtitle {
  font-size: 0.95rem;
  color: var(--text-muted);
  margin: 0;
}
.ig-page-header-right {
  display: flex;
  align-items: center;
  gap: 1rem;
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

/* ── Professional Cards ── */
.ig-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 1.5rem;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s, border-color 0.2s;
  height: 100%;
}
.ig-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-light);
}
.ig-card-title {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  font-weight: 600;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.ig-card-value {
  font-size: 1.875rem;
  font-weight: 700;
  color: var(--text-main);
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.ig-card-trend {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}
.ig-trend-up { color: var(--success); }
.ig-trend-down { color: var(--danger); }
.ig-trend-neutral { color: var(--text-light); }

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

/* ── Sidebar (App Shell) ── */
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
  margin-bottom: 1.5rem !important;
}
/* Style the radio buttons to look like nav links */
[data-testid="stSidebar"] .stRadio label {
  cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  padding: 0.5rem 0.75rem;
  border-radius: var(--r-sm);
  margin-bottom: 0.1rem;
  transition: background 0.15s, color 0.15s;
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
/* Selected state for radio */
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div {
  background: var(--primary-s);
  border-radius: var(--r-sm);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-baseweb="radio"] input:checked + div p {
  color: #60a5fa !important;
}
/* Hide the actual radio circle */
[data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] div:first-child {
  display: none !important;
}

/* ── Custom Data Tables ── */
.stDataFrame {
  border-radius: var(--r-md) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}
.stDataFrame [data-testid="StyledFullScreenButton"] {
  display: none !important;
}

/* ── Customer 360 Header ── */
.ig-c360-header {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 1.5rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  margin-bottom: 1.5rem;
}
.ig-c360-avatar {
  width: 64px;
  height: 64px;
  background: var(--surface-hover);
  border-radius: var(--r-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  border: 1px solid var(--border-light);
}
.ig-c360-info { flex: 1; }
.ig-c360-name {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-main);
  margin: 0 0 0.5rem 0;
  display: flex;
  align-items: center;
  gap: 1rem;
}
.ig-c360-meta {
  display: flex;
  gap: 1.5rem;
  color: var(--text-muted);
  font-size: 0.9rem;
}
.ig-c360-meta span {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

/* ── Recommendation Callouts ── */
.ig-recommendation {
  background: var(--surface);
  border-left: 4px solid var(--primary);
  padding: 1.25rem;
  border-radius: 0 var(--r-md) var(--r-md) 0;
  margin-bottom: 1rem;
}
.ig-recommendation-title {
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.ig-recommendation-body {
  color: var(--text-light);
  font-size: 0.95rem;
  line-height: 1.5;
}

/* ── Insight Signals ── */
.ig-signal {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
  padding: 0.75rem 0;
  border-bottom: 1px solid var(--border);
}
.ig-signal:last-child { border-bottom: none; }
.ig-signal-icon { font-size: 1.1rem; margin-top: 0.1rem; }
.ig-signal-content { flex: 1; }
.ig-signal-title { font-weight: 500; color: var(--text-main); margin-bottom: 0.25rem; }
.ig-signal-desc { font-size: 0.85rem; color: var(--text-muted); }

/* ── Utility Classes ── */
.ig-text-mono { font-family: var(--font-mono); }
.ig-empty-state {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
  background: var(--surface);
  border: 1px dashed var(--border-light);
  border-radius: var(--r-md);
}
.ig-empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.5; }
.ig-empty-title { font-weight: 600; font-size: 1.1rem; color: var(--text-main); margin-bottom: 0.5rem; }
</style>
        """,
        unsafe_allow_html=True,
    )
