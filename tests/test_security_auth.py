import os
import unittest
import streamlit as st
import pandas as pd
from unittest.mock import patch, MagicMock

from src.security.auth import (
    ROLE_USER,
    ROLE_ADMIN,
    hash_password,
    verify_password,
    authenticate_admin,
    login_admin,
    logout,
    is_admin,
    get_current_role,
    require_admin,
    is_production_environment,
    is_demo_fallback_allowed,
)
from src.ui.navigation import get_nav_groups, _BUSINESS_NAV_GROUPS, _ADMIN_NAV_GROUP
from src.data.file_loader import validate_and_load_uploaded_file, MAX_FILE_SIZE_MB


class MockUpload:
    def __init__(self, name: str, data: bytes, size: int = None):
        self.name = name
        self.data = data
        self.size = len(data) if size is None else size
        import io
        self._io = io.BytesIO(data)

    def seek(self, offset):
        self._io.seek(offset)

    def read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)


class TestSecurityAuth(unittest.TestCase):
    def setUp(self):
        # Reset session state before each test
        st.session_state["auth_role"] = ROLE_USER
        st.session_state["auth_user"] = None
        st.session_state["auth_authenticated"] = False
        st.session_state["current_page"] = "Overview"

    def tearDown(self):
        st.session_state["auth_role"] = ROLE_USER
        st.session_state["auth_user"] = None
        st.session_state["auth_authenticated"] = False

    def test_password_hashing_and_verification(self):
        """Verify PBKDF2 password hashing and constant-time verification."""
        hashed = hash_password("SuperSecretPass123!", salt="testsalt123")
        self.assertIn("$", hashed)
        self.assertTrue(verify_password("SuperSecretPass123!", hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))
        self.assertFalse(verify_password("", hashed))
        self.assertFalse(verify_password("SuperSecretPass123!", ""))

    def test_default_admin_authentication_in_local_dev(self):
        """Verify fallback demo admin credentials work only in non-production local dev."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "INVOICEGUARD_ENV": "development"}):
            self.assertTrue(authenticate_admin("admin", "admin123"))
            self.assertFalse(authenticate_admin("admin", "wrongpassword"))
            self.assertFalse(authenticate_admin("unknown_user", "admin123"))
            self.assertFalse(authenticate_admin("", ""))

    def test_production_strictly_rejects_fallback_demo_credentials(self):
        """Verify that production rejects fallback demo credentials if no explicit secrets are configured."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "INVOICEGUARD_ADMIN_USERNAME": "",
            "INVOICEGUARD_ADMIN_PASSWORD": "",
            "INVOICEGUARD_ADMIN_PASSWORD_HASH": "",
        }):
            self.assertTrue(is_production_environment())
            self.assertFalse(is_demo_fallback_allowed())
            # In production without explicit credentials, demo credentials must be denied
            self.assertFalse(authenticate_admin("admin", "admin123"))

    def test_production_accepts_explicit_credentials(self):
        """Verify production permits authentication only when explicit credentials are configured."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "INVOICEGUARD_ADMIN_USERNAME": "cloud_admin",
            "INVOICEGUARD_ADMIN_PASSWORD": "CloudAdminSecret!999",
        }):
            self.assertTrue(is_production_environment())
            self.assertTrue(authenticate_admin("cloud_admin", "CloudAdminSecret!999"))
            # Demo credentials must still fail even when explicit ones exist
            self.assertFalse(authenticate_admin("admin", "admin123"))
            self.assertFalse(authenticate_admin("cloud_admin", "wrongpassword"))

    def test_custom_environment_credentials(self):
        """Verify credentials configured via environment variables."""
        with patch.dict(os.environ, {
            "INVOICEGUARD_ADMIN_USERNAME": "secops_admin",
            "INVOICEGUARD_ADMIN_PASSWORD": "CustomAdminSecret!456",
        }):
            self.assertTrue(authenticate_admin("secops_admin", "CustomAdminSecret!456"))
            self.assertFalse(authenticate_admin("admin", "admin123"))
            self.assertFalse(authenticate_admin("secops_admin", "badpassword"))

    def test_hashed_environment_credentials(self):
        """Verify hashed password authentication via environment variable."""
        hashed_pw = hash_password("HashProtectedSecret!789", salt="envsalt")
        with patch.dict(os.environ, {
            "INVOICEGUARD_ADMIN_USERNAME": "hashed_admin",
            "INVOICEGUARD_ADMIN_PASSWORD_HASH": hashed_pw,
        }):
            self.assertTrue(authenticate_admin("hashed_admin", "HashProtectedSecret!789"))
            self.assertFalse(authenticate_admin("hashed_admin", "wrong_attempt"))

    def test_session_role_lifecycle(self):
        """Test guest -> admin -> logout -> guest lifecycle."""
        # Initial guest/user state
        self.assertEqual(get_current_role(), ROLE_USER)
        self.assertFalse(is_admin())

        # Successful admin login
        success = login_admin("admin", "admin123")
        self.assertTrue(success)
        self.assertEqual(get_current_role(), ROLE_ADMIN)
        self.assertTrue(is_admin())

        # Page switching while authenticated preserves admin role
        st.session_state["current_page"] = "Customer Risk"
        self.assertTrue(is_admin())
        st.session_state["current_page"] = "System Status"
        self.assertTrue(is_admin())

        # Admin logout clears credentials and redirects safely
        logout()
        self.assertEqual(get_current_role(), ROLE_USER)
        self.assertFalse(is_admin())
        self.assertIsNone(st.session_state.get("auth_user"))
        self.assertEqual(st.session_state.get("current_page"), "Overview")

    def test_session_tampering_resilience(self):
        """Verify is_admin() requires both authenticated flag AND admin role."""
        st.session_state["auth_role"] = ROLE_ADMIN
        st.session_state["auth_authenticated"] = False
        self.assertFalse(is_admin())

    def test_require_admin_guard(self):
        """Verify require_admin blocks unauthenticated users and allows admins."""
        st.session_state["auth_role"] = ROLE_USER
        st.session_state["auth_authenticated"] = False
        self.assertFalse(require_admin())

        st.session_state["auth_role"] = ROLE_ADMIN
        st.session_state["auth_authenticated"] = True
        self.assertTrue(require_admin())

    def test_navigation_role_filtering(self):
        """Verify navigation groups hide admin views from normal users."""
        st.session_state["auth_role"] = ROLE_USER
        st.session_state["auth_authenticated"] = False
        user_nav = get_nav_groups()
        group_names = [name for name, _ in user_nav]
        all_pages = [page for _, pages in user_nav for page in pages]

        self.assertNotIn("ADMIN", group_names)
        self.assertNotIn("Data Quality", all_pages)
        self.assertNotIn("System Status", all_pages)
        self.assertEqual(len(user_nav), 5)

        # As Admin
        st.session_state["auth_role"] = ROLE_ADMIN
        st.session_state["auth_authenticated"] = True
        admin_nav = get_nav_groups()
        admin_group_names = [name for name, _ in admin_nav]
        admin_pages = [page for _, pages in admin_nav for page in pages]

        self.assertIn("ADMIN", admin_group_names)
        self.assertIn("Data Quality", admin_pages)
        self.assertIn("System Status", admin_pages)
        self.assertEqual(len(admin_nav), 6)

    def test_env_example_contains_placeholders_only(self):
        """Verify .env.example does not contain real secrets or production credentials."""
        env_example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
        if os.path.exists(env_example_path):
            with open(env_example_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("change_this_in_production", content)
            self.assertNotIn("supersecret", content.lower())

    def test_file_upload_empty_file_rejected(self):
        """Verify 0-byte upload is safely handled."""
        empty_upload = MockUpload("test.csv", b"", size=0)
        df, err = validate_and_load_uploaded_file(empty_upload)
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

    def test_file_upload_oversized_rejected(self):
        """Verify oversized files (>50MB) are safely handled."""
        oversized = MockUpload("big.csv", b"sample", size=55 * 1024 * 1024)
        df, err = validate_and_load_uploaded_file(oversized)
        self.assertIsNone(df)
        self.assertIn("exceeds maximum allowed", err)

    def test_file_upload_unsupported_extension_rejected(self):
        """Verify unsupported file extensions are safely handled."""
        bad_ext = MockUpload("script.exe", b"binary content")
        df, err = validate_and_load_uploaded_file(bad_ext)
        self.assertIsNone(df)
        self.assertIn("unsupported file format", err.lower())

    def test_file_upload_error_sanitization(self):
        """Verify that file load parsing errors never leak internal filesystem paths."""
        corrupted = MockUpload("malformed.csv", b"\x00\xff\xfe\x12\x34\x56")
        df, err = validate_and_load_uploaded_file(corrupted)
        if err:
            self.assertNotIn("C:\\", err)
            self.assertNotIn("/Users/", err)
            self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
