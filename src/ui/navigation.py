import streamlit as st
from typing import Dict, Any, List, Tuple
from src.security.auth import is_admin, login_admin, logout

_BUSINESS_NAV_GROUPS: List[Tuple[str, List[str]]] = [
    ("DASHBOARD",    ["Overview"]),
    ("RECEIVABLES",  ["Invoices", "Collections"]),
    ("CUSTOMERS",    ["Customer 360", "Customer Search", "Customer Risk", "Customer Segmentation"]),
    ("INTELLIGENCE", ["Risk Drivers", "Message Intelligence", "Product Intelligence"]),
    ("REVENUE",      ["Revenue Forecast", "Customer Retention", "Revenue at Risk"]),
]

_ADMIN_NAV_GROUP: Tuple[str, List[str]] = ("ADMIN", ["Data Quality", "System Status"])


def get_nav_groups() -> List[Tuple[str, List[str]]]:
    """Returns active navigation groups filtered by authorization level."""
    groups = list(_BUSINESS_NAV_GROUPS)
    if is_admin():
        groups.append(_ADMIN_NAV_GROUP)
    return groups


def render_top_header(current_page: str) -> None:
    """
    Renders the enterprise application top navigation shell with branding,
    hierarchical breadcrumb, theme toggle, and live authorization status.
    """
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    current_theme = st.session_state.theme

    # Determine parent navigation group
    section = "WORKSPACE"
    for group_name, pages in get_nav_groups():
        if current_page in pages:
            section = group_name
            break

    is_dark = (current_theme == "dark")
    toggle_icon = "☀️ Light" if is_dark else "🌙 Dark"

    c_left, c_right = st.columns([3.2, 1.2])
    with c_left:
        st.markdown(
            f"""
            <div class="ig-shell-bar">
                <div class="ig-shell-brand-group">
                    <span class="ig-shell-logo">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                    </span>
                    <span class="ig-shell-title">InvoiceGuard AI</span>
                    <span class="ig-shell-sep">/</span>
                    <span class="ig-shell-section">{section}</span>
                    <span class="ig-shell-sep">/</span>
                    <span class="ig-shell-page">{current_page}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_right:
        col_btn, col_pill = st.columns([1.1, 1.1])
        with col_btn:
            if st.button(toggle_icon, key="top_theme_toggle_btn", use_container_width=True, type="secondary"):
                st.session_state.theme = "light" if is_dark else "dark"
                st.rerun()
        with col_pill:
            role_text = "Admin" if is_admin() else "Analyst"
            st.markdown(
                f"""
                <div class="ig-shell-role-container">
                    <span class="ig-shell-role-pill">
                        <span class="ig-shell-status-dot"></span>
                        {role_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar() -> str:
    """Renders the enterprise sidebar with deterministic active-state styling and returns the selected page."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    with st.sidebar:
        st.markdown(
            """
            <div class="ig-sidebar-header" style="padding-top:0.5rem; margin-bottom:1.5rem;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
                    <span style="color:var(--primary); display:flex; align-items:center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                    </span>
                    <span style="color:var(--text-main); font-size:1.25rem; font-weight:700; letter-spacing:-0.02em;">
                        InvoiceGuard <span style="color:var(--primary); font-weight:800;">AI</span>
                    </span>
                </div>
                <p style="color:var(--text-muted); font-size:0.75rem; margin:0; font-weight:500;">Enterprise Receivables Intelligence</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        nav_groups = get_nav_groups()

        for group_name, pages in nav_groups:
            st.markdown(
                f'<p class="ig-sidebar-group-label" style="font-size:0.68rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin:1.25rem 0 0.4rem 0.2rem;">{group_name}</p>', 
                unsafe_allow_html=True
            )
            
            for page in pages:
                is_active = (st.session_state.current_page == page)
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(page, key=f"nav_{page}", use_container_width=True, type=btn_type):
                    if st.session_state.current_page != page:
                        st.session_state.current_page = page
                        st.rerun()

        # Operational Footer & Theme Quick Toggle
        st.markdown(
            """
            <hr style="border:none; border-top:1px solid var(--border); margin:1.75rem 0 1rem 0;"/>
            <div style="font-size:0.75rem; color:var(--text-muted); display:flex; align-items:center; justify-content:space-between; padding:0 0.2rem; margin-bottom:0.75rem;">
                <span>v3.0 Production</span>
                <span style="color:var(--success); font-weight:600; display:inline-flex; align-items:center; gap:0.35rem;">
                    <span style="width:6px; height:6px; border-radius:50%; background:var(--success); display:inline-block;"></span>
                    Operational
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # Sidebar Theme Toggle
        theme = st.session_state.theme
        theme_btn_text = "☀️ Theme: Light" if theme == "dark" else "🌙 Theme: Dark"
        if st.button(theme_btn_text, key="sidebar_theme_toggle_btn", use_container_width=True, type="secondary"):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()

        # Admin Auth Panel
        if is_admin():
            st.markdown(
                """
                <div style="margin-top:0.75rem; padding:0.5rem 0.65rem; background:var(--primary-s); border:1px solid var(--primary); border-radius:var(--r-sm); display:flex; align-items:center; justify-content:space-between;">
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
                🔒 <strong>Enterprise Demo:</strong> In-memory processing with role-based access control and restricted diagnostic surfaces.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return str(st.session_state.current_page)
