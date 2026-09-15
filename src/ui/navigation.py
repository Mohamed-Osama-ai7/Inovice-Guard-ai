import streamlit as st
from typing import Dict, Any

_NAV_GROUPS = [
    ("DASHBOARD",    ["Overview"]),
    ("RECEIVABLES",  ["Invoices", "Collections"]),
    ("CUSTOMERS",    ["Customer 360", "Customer Search", "Customer Risk"]),
    ("INTELLIGENCE", ["Risk Drivers", "Message Intelligence"]),
    ("REVENUE",      ["Revenue Forecast", "Customer Retention", "Revenue at Risk"]),
]


def render_sidebar() -> str:
    """Renders the enterprise sidebar with deterministic active-state styling and returns the selected page."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    with st.sidebar:
        st.markdown(
            """
            <div style="padding-top:0.5rem; margin-bottom:1.5rem;">
                <h1 style="color:var(--primary) !important; font-size:1.35rem !important; display:flex; align-items:center; gap:0.5rem; margin-bottom:0 !important; font-weight:700 !important;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                    InvoiceGuard AI
                </h1>
                <p style="color:var(--text-muted); font-size:0.75rem; margin-top:0.25rem; margin-bottom:0; font-weight:500;">Receivables & Risk Intelligence</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        for group_name, pages in _NAV_GROUPS:
            st.markdown(
                f'<p style="font-size:0.68rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin:1.2rem 0 0.4rem 0.2rem;">{group_name}</p>', 
                unsafe_allow_html=True
            )
            
            for page in pages:
                is_active = (st.session_state.current_page == page)
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(page, key=f"nav_{page}", use_container_width=True, type=btn_type):
                    if st.session_state.current_page != page:
                        st.session_state.current_page = page
                        st.rerun()

        # Operational Footer
        st.markdown(
            """
            <hr style="border-color:var(--border); margin:2rem 0 1rem 0;"/>
            <div style="font-size:0.75rem; color:var(--text-muted); display:flex; align-items:center; justify-content:space-between; padding:0 0.2rem;">
                <span>InvoiceGuard v3.0</span>
                <span style="color:var(--success); font-weight:600; display:inline-flex; align-items:center; gap:0.35rem;">
                    <span style="width:6px; height:6px; border-radius:50%; background:var(--success); display:inline-block;"></span>
                    System Operational
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    return str(st.session_state.current_page)
