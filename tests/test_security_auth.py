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

    def test_default_admin_authentication(self):
        """Verify default demo admin credentials work in test environment."""
        self.assertTrue(authenticate_admin("admin", "admin123"))
        self.assertFalse(authenticate_admin("admin", "wrongpassword"))
        self.assertFalse(authenticate_admin("unknown_user", "admin123"))
        self.assertFalse(authenticate_admin("", ""))

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
        """Test login_admin elevates role and logout safely demotes to USER."""
        self.assertEqual(get_current_role(), ROLE_USER)
        self.assertFalse(is_admin())

        success = login_admin("admin", "admin123")
        self.assertTrue(success)
        self.assertEqual(get_current_role(), ROLE_ADMIN)
        self.assertTrue(is_admin())

        # When logged in as admin viewing an admin page
        st.session_state["current_page"] = "System Status"
        logout()
        self.assertEqual(get_current_role(), ROLE_USER)
        self.assertFalse(is_admin())
        self.assertEqual(st.session_state["current_page"], "Overview")

    def test_require_admin_guard(self):
        """Verify require_admin blocks unauthenticated users and allows admins."""
        st.session_state["auth_role"] = ROLE_USER
        self.assertFalse(require_admin())

        st.session_state["auth_role"] = ROLE_ADMIN
        self.assertTrue(require_admin())

    def test_navigation_role_filtering(self):
        """Verify navigation groups hide admin views from normal users."""
        st.session_state["auth_role"] = ROLE_USER
        user_nav = get_nav_groups()
        group_names = [name for name, _ in user_nav]
        all_pages = [page for _, pages in user_nav for page in pages]

        self.assertNotIn("ADMIN", group_names)
        self.assertNotIn("Data Quality", all_pages)
        self.assertNotIn("System Status", all_pages)
        self.assertEqual(len(user_nav), 5)

        # As Admin
        st.session_state["auth_role"] = ROLE_ADMIN
        admin_nav = get_nav_groups()
        admin_group_names = [name for name, _ in admin_nav]
        admin_pages = [page for _, pages in admin_nav for page in pages]

        self.assertIn("ADMIN", admin_group_names)
        self.assertIn("Data Quality", admin_pages)
        self.assertIn("System Status", admin_pages)
        self.assertEqual(len(admin_nav), 6)

    def test_file_upload_empty_file_rejected(self):
        """Verify 0-byte upload is safely rejected."""
        empty_upload = MockUpload("test.csv", b"", size=0)
        df, err = validate_and_load_uploaded_file(empty_upload)
        self.assertIsNone(df)
        self.assertIn("empty", err.lower())

    def test_file_upload_oversized_rejected(self):
        """Verify oversized files (>50MB) are safely rejected."""
        oversized = MockUpload("big.csv", b"sample", size=55 * 1024 * 1024)
        df, err = validate_and_load_uploaded_file(oversized)
        self.assertIsNone(df)
        self.assertIn("exceeds maximum allowed", err)

    def test_file_upload_unsupported_extension_rejected(self):
        """Verify unsupported file extensions are safely rejected."""
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
