import streamlit as st
from typing import Dict, Any

_NAV_GROUPS = [
    ("OVERVIEW",            ["Overview"]),
    ("RECEIVABLES",         ["Invoice Risk", "Collections"]),
    ("CUSTOMER INTELLIGENCE",["Customer 360", "Customer Search", "Customer Risk"]),
    ("REVENUE INTELLIGENCE",["Revenue Forecast", "Repurchase Risk", "Revenue at Risk"]),
    ("AI INSIGHTS",         ["AI Recommendations", "Risk Signals", "Explainable AI"]),
    ("SYSTEM",              ["Model Performance", "Data Quality", "System Health"]),
]


def render_sidebar() -> str:
    """Renders the enterprise sidebar and returns the selected page."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    with st.sidebar:
        st.markdown(
            """
            <div style="padding-top:0.5rem;">
                <h1 style="color:var(--primary) !important; display:flex; align-items:center; gap:0.5rem; margin-bottom:0 !important;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                    InvoiceGuard AI
                </h1>
                <p style="color:var(--text-muted); font-size:0.8rem; margin-bottom:2rem; font-weight:500;">Enterprise Intelligence Platform</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        for group_name, pages in _NAV_GROUPS:
            st.markdown(f'<p style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin:1.5rem 0 0.5rem 0;">{group_name}</p>', unsafe_allow_html=True)
            
            # Find index if current page is in this group
            idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else None
            
            # We use radio. Streamlit will reset other radios if we don't manage them, 
            # but since we want it to look good, we can use session state callbacks.
            selected = st.radio(
                f"{group_name}_nav",
                options=pages,
                index=idx if idx is not None else None,
                label_visibility="collapsed",
                key=f"radio_{group_name}"
            )
            
            if selected and selected != st.session_state.current_page:
                st.session_state.current_page = selected
                st.rerun()

        # Footer
        st.markdown('<div style="flex-grow:1;"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <hr style="border-color:var(--border); margin-top:3rem;"/>
            <div style="font-size:0.75rem; color:var(--text-muted); display:flex; justify-content:space-between;">
                <span>v3.0.0 Enterprise</span>
                <span style="color:var(--success);">● Operational</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    return st.session_state.current_page
