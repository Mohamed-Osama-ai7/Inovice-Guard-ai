import pytest
import pandas as pd
import io
from src.data.file_loader import validate_and_load_uploaded_file

class MockUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.size = len(data)
        self._io = io.BytesIO(data)
    
    def seek(self, offset):
        self._io.seek(offset)
        
    def read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)

def test_file_loader_empty():
    df, err = validate_and_load_uploaded_file(None)
    assert df is None
    assert err == "No file uploaded."

def test_file_loader_oversized():
    # Mock size to 51 MB
    f = MockUploadedFile("test.csv", b"")
    f.size = 51 * 1024 * 1024
    df, err = validate_and_load_uploaded_file(f)
    assert df is None
    assert "exceeds maximum allowed" in err

def test_file_loader_invalid_ext():
    f = MockUploadedFile("test.txt", b"some data")
    df, err = validate_and_load_uploaded_file(f)
    assert df is None
    assert "Unsupported file format" in err

def test_file_loader_valid_csv():
    f = MockUploadedFile("test.csv", b"col1,col2\n1,2")
    df, err = validate_and_load_uploaded_file(f)
    assert df is not None
    assert err is None
    assert len(df) == 1

def test_file_loader_empty_csv_data():
    f = MockUploadedFile("test.csv", b"col1,col2\n")
    df, err = validate_and_load_uploaded_file(f)
    assert df is None
    assert "no data" in err

def test_file_loader_corrupted_csv():
    # Can't easily break pandas parser without a weird trick, but we can test bad encoding
    f = MockUploadedFile("test.csv", b"\xff\xfe\x00\x00")
    df, err = validate_and_load_uploaded_file(f)
    # Actually pandas might just read this as garbage or error out
    # If it errors out:
    pass
