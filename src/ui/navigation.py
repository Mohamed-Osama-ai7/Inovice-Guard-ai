import streamlit as st
from typing import Dict, Any, List, Tuple
from src.security.auth import is_admin, login_admin, logout

_BUSINESS_NAV_GROUPS: List[Tuple[str, List[str]]] = [
    ("DASHBOARD",    ["Overview"]),
    ("RECEIVABLES",  ["Invoices", "Collections"]),
    ("CUSTOMERS",    ["Customer 360", "Customer Search", "Customer Risk"]),
    ("INTELLIGENCE", ["Risk Drivers", "Message Intelligence"]),
    ("REVENUE",      ["Revenue Forecast", "Customer Retention", "Revenue at Risk"]),
]

_ADMIN_NAV_GROUP: Tuple[str, List[str]] = ("ADMIN", ["Data Quality", "System Status"])


def get_nav_groups() -> List[Tuple[str, List[str]]]:
    """Returns active navigation groups filtered by authorization level."""
    groups = list(_BUSINESS_NAV_GROUPS)
    if is_admin():
        groups.append(_ADMIN_NAV_GROUP)
    return groups


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

        nav_groups = get_nav_groups()

        for group_name, pages in nav_groups:
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
            <hr style="border-color:var(--border); margin:1.75rem 0 1rem 0;"/>
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

        # Admin Auth Panel
        if is_admin():
            st.markdown(
                """
                <div style="margin-top:0.75rem; padding:0.5rem 0.65rem; background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.25); border-radius:4px; display:flex; align-items:center; justify-content:space-between;">
                    <span style="font-size:0.75rem; font-weight:600; color:var(--primary); display:flex; align-items:center; gap:0.35rem;">
                        🛡️ Admin Active
                    </span>
                    <span style="font-size:0.68rem; color:var(--text-muted);">Privileged</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Sign Out", key="admin_logout_btn", use_container_width=True, type="secondary"):
                logout()
                st.rerun()
        else:
            with st.expander("Admin Access", expanded=False):
                with st.form("admin_login_form", clear_on_submit=True):
                    admin_u = st.text_input("Username", key="admin_user_input", placeholder="Admin ID")
                    admin_p = st.text_input("Password", type="password", key="admin_pwd_input", placeholder="Password")
                    submitted = st.form_submit_button("Authenticate", use_container_width=True)
                    if submitted:
                        if login_admin(admin_u, admin_p):
                            st.success("Authenticated.")
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")

        # Demo Privacy Notice
        st.markdown(
            """
            <div style="font-size:0.68rem; color:var(--text-muted); margin-top:1rem; line-height:1.35; opacity:0.75; padding:0 0.2rem;">
                🔒 <strong>Demo Notice:</strong> Do not upload confidential, personally identifiable, or production customer data. Client data is processed in-memory with controlled access, validated file handling, and restricted administrative diagnostics.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return str(st.session_state.current_page)
