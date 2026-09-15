"""InvoiceGuard AI - Security & Authentication Package."""
from src.security.auth import (
    ROLE_USER,
    ROLE_ADMIN,
    get_current_role,
    is_admin,
    require_admin,
    login_admin,
    logout,
    hash_password,
    verify_password,
    authenticate_admin,
    is_production_environment,
    is_demo_fallback_allowed,
)

__all__ = [
    "ROLE_USER",
    "ROLE_ADMIN",
    "get_current_role",
    "is_admin",
    "require_admin",
    "login_admin",
    "logout",
    "hash_password",
    "verify_password",
    "authenticate_admin",
    "is_production_environment",
    "is_demo_fallback_allowed",
]
