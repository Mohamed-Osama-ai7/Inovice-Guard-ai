"""
InvoiceGuard AI - Authentication & Authorization Security Module
Enforces role-based access control (RBAC) separating public/business users
from administrative and diagnostic capabilities.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

logger = logging.getLogger("invoiceguard.security")

ROLE_USER = "USER"
ROLE_ADMIN = "ADMIN"

# Session state keys
SESSION_ROLE_KEY = "auth_role"
SESSION_USER_KEY = "auth_user"
SESSION_AUTH_KEY = "auth_authenticated"

# Default fallback values for local/demo testing only (strictly forbidden in production)
_DEFAULT_ADMIN_USER = "admin"
_DEFAULT_ADMIN_PASS = "admin123"
_DEFAULT_SALT = "invoiceguard_salt_2026"


def is_production_environment() -> bool:
    """
    Determines if the application is running in a production environment
    (e.g., Streamlit Community Cloud, production container, or explicit env var).
    """
    # 1. Explicit production environment variable
    env = os.environ.get("INVOICEGUARD_ENV", os.environ.get("ENVIRONMENT", "")).lower().strip()
    if env in ("production", "prod"):
        return True

    # 2. Explicit demo mode override (if set to false/0, treat as production/restricted)
    allow_demo = os.environ.get("INVOICEGUARD_ALLOW_DEMO_LOGIN", "").lower().strip()
    if allow_demo in ("0", "false", "no", "disabled"):
        return True

    # 3. Streamlit Community Cloud runtime indicators
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("IS_STREAMLIT_CLOUD"):
        return True

    # 4. Hosted cloud file paths (e.g. /mount/src on Streamlit Cloud or /app/src)
    file_path = str(os.path.abspath(__file__)).replace("\\", "/")
    if "/mount/src/" in file_path or "/app/src/" in file_path:
        return True

    return False


def is_demo_fallback_allowed() -> bool:
    """
    Returns True only in non-production local development when demo fallback is permitted.
    Production strictly rejects any fallback demo credentials.
    """
    return not is_production_environment()


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Computes a salted PBKDF2-HMAC-SHA256 hash formatted as salt$hexdigest."""
    if salt is None:
        salt = _DEFAULT_SALT
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash_or_plain: str) -> bool:
    """
    Verifies a password against a stored PBKDF2 hash (salt$hexdigest)
    or against a plaintext secret using constant-time comparison.
    """
    if not password or not stored_hash_or_plain:
        return False

    if "$" in stored_hash_or_plain:
        try:
            salt, expected_hex = stored_hash_or_plain.split("$", 1)
            dk = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                iterations=100_000,
            )
            return hmac.compare_digest(dk.hex(), expected_hex)
        except Exception as exc:
            logger.warning("Error verifying hashed password: %s", exc)
            return False

    # Plaintext fallback (e.g. standard environment variable)
    return hmac.compare_digest(
        password.encode("utf-8"),
        stored_hash_or_plain.encode("utf-8"),
    )


def _get_config(key: str, default: str = "") -> str:
    """
    Resolves configuration prioritizing st.secrets then os.environ,
    with a safe fallback default.
    """
    # Try st.secrets first if available
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key in st.secrets:
                return str(st.secrets[key])
            # Check nested sections
            admin_sec = st.secrets.get("admin", {})
            if isinstance(admin_sec, dict) and key in admin_sec:
                return str(admin_sec[key])
            auth_sec = st.secrets.get("auth", {})
            if isinstance(auth_sec, dict) and key in auth_sec:
                return str(auth_sec[key])
    except Exception:
        pass

    # Next check os.environ
    return os.environ.get(key, default)


def authenticate_admin(username: str, password: str) -> bool:
    """
    Validates administrator credentials using constant-time comparisons.
    In production, explicit credentials configured via st.secrets or environment
    variables are mandatory; fallback demo credentials are never accepted.
    In local development, fallback demo credentials (admin / admin123) are permitted
    only if no explicit credentials have been configured.
    """
    if not username or not password:
        return False

    explicit_user = _get_config("INVOICEGUARD_ADMIN_USERNAME", default="")
    explicit_pass = _get_config("INVOICEGUARD_ADMIN_PASSWORD", default="")
    explicit_hash = _get_config("INVOICEGUARD_ADMIN_PASSWORD_HASH", default="")

    has_explicit_creds = bool(explicit_user and (explicit_pass or explicit_hash))

    if not has_explicit_creds:
        if not is_demo_fallback_allowed():
            logger.warning(
                "Admin authentication denied: Production environment requires explicit "
                "admin credentials configured via st.secrets or environment variables."
            )
            return False
        # Local development demo fallback only
        configured_user = _DEFAULT_ADMIN_USER
        configured_pass = _DEFAULT_ADMIN_PASS
        configured_hash = ""
    else:
        configured_user = explicit_user
        configured_pass = explicit_pass
        configured_hash = explicit_hash

    # Constant-time username match
    user_match = hmac.compare_digest(
        username.strip().encode("utf-8"),
        configured_user.strip().encode("utf-8"),
    )
    if not user_match:
        logger.warning("Admin authentication failed: unrecognized username.")
        return False

    # Constant-time password verification
    if configured_hash:
        pass_valid = verify_password(password, configured_hash)
    else:
        pass_valid = verify_password(password, configured_pass)

    if not pass_valid:
        logger.warning("Admin authentication failed: invalid credentials provided.")
        return False

    logger.info("Admin authentication successful for user: %s", username.strip())
    return True


def get_current_role() -> str:
    """Returns the current session role (defaults to ROLE_USER)."""
    try:
        import streamlit as st
        return st.session_state.get(SESSION_ROLE_KEY, ROLE_USER)
    except Exception:
        return ROLE_USER


def is_admin() -> bool:
    """Returns True if the active session is authenticated with ADMIN role."""
    try:
        import streamlit as st
        is_auth = bool(st.session_state.get(SESSION_AUTH_KEY, False))
        role = st.session_state.get(SESSION_ROLE_KEY, ROLE_USER)
        return is_auth and (role == ROLE_ADMIN)
    except Exception:
        return False


def login_admin(username: str, password: str) -> bool:
    """
    Validates admin credentials and elevates session role to ADMIN on success.
    Returns True if login succeeded, False otherwise.
    """
    if authenticate_admin(username, password):
        try:
            import streamlit as st
            st.session_state[SESSION_ROLE_KEY] = ROLE_ADMIN
            st.session_state[SESSION_USER_KEY] = username.strip()
            st.session_state[SESSION_AUTH_KEY] = True
        except Exception:
            pass
        return True
    return False


def logout() -> None:
    """Clears administrator credentials and reverts session role to ROLE_USER."""
    try:
        import streamlit as st
        st.session_state[SESSION_ROLE_KEY] = ROLE_USER
        st.session_state[SESSION_USER_KEY] = None
        st.session_state[SESSION_AUTH_KEY] = False
        # If user was viewing an admin page, redirect safely to Overview
        if st.session_state.get("current_page") in ("Data Quality", "System Status"):
            st.session_state.current_page = "Overview"
    except Exception:
        pass


def require_admin() -> bool:
    """
    Server-side authorization enforcement guard.
    Returns True if the current session has ADMIN authorization.
    If unauthorized, logs an audit warning, displays a clean access-restricted
    card in the UI, and returns False to halt execution of the calling view.
    """
    if is_admin():
        return True

    logger.warning("Unauthorized access attempt to administrative diagnostic view.")
    try:
        import streamlit as st
        st.markdown(
            """
            <div class="ig-card" style="border-left: 4px solid var(--danger); margin-top: 1rem; margin-bottom: 1.5rem;">
              <div class="ig-card-title" style="color: var(--danger); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; display:flex; align-items:center; gap:0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                Access Restricted
              </div>
              <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-main); margin-bottom: 0.35rem;">
                Administrator Privileges Required
              </div>
              <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0; line-height: 1.4;">
                This diagnostic resource is restricted to authorized platform administrators.
                Please sign in with verified administrator credentials via the Admin Access panel in the sidebar.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass
    return False
