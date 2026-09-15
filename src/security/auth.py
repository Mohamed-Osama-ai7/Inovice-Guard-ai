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

# Default fallback values for local/demo testing (overridden by env vars or st.secrets in production)
_DEFAULT_ADMIN_USER = "admin"
_DEFAULT_ADMIN_PASS = "admin123"
_DEFAULT_SALT = "invoiceguard_salt_2026"


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
    Returns True if valid, False otherwise.
    """
    if not username or not password:
        return False

    configured_user = _get_config("INVOICEGUARD_ADMIN_USERNAME", _DEFAULT_ADMIN_USER)
    
    # Constant-time username match
    user_match = hmac.compare_digest(
        username.strip().encode("utf-8"),
        configured_user.strip().encode("utf-8"),
    )
    if not user_match:
        logger.warning("Admin authentication failed: unrecognized username.")
        return False

    # Check for password hash or plaintext secret
    configured_hash = _get_config("INVOICEGUARD_ADMIN_PASSWORD_HASH", "")
    if configured_hash:
        pass_valid = verify_password(password, configured_hash)
    else:
        configured_pass = _get_config("INVOICEGUARD_ADMIN_PASSWORD", _DEFAULT_ADMIN_PASS)
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
    return get_current_role() == ROLE_ADMIN


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
