"""
Tests for database functionality.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from llm_consortium import DatabaseConnection, log_response, logs_db_path

@pytest.fixture
def mock_db():
    with patch('sqlite_utils.Database') as mock:
        yield mock

@pytest.fixture
def mock_response():
    response = Mock()
    response.text.return_value = "Test response"
    response.model = "test-model"
    return response

def test_database_connection_singleton(mock_db):
    """Test that DatabaseConnection maintains singleton pattern."""
    db1 = DatabaseConnection.get_connection()
    db2 = DatabaseConnection.get_connection()
    assert db1 is db2

def test_logs_db_path():
    """Test logs database path generation."""
    path = logs_db_path()
    assert isinstance(path, Path)
    assert str(path).endswith('logs.db')

def test_log_response(mock_db, mock_response):
    """Test logging responses to database."""
    with patch('llm_consortium.DatabaseConnection.get_connection') as mock_get_conn:
        mock_get_conn.return_value = mock_db
        log_response(mock_response, "test-model")
        mock_response.log_to_db.assert_called_once_with(mock_db)

def test_log_response_error_handling(mock_db, mock_response):
    """Test error handling during response logging."""
    mock_response.log_to_db.side_effect = Exception("Test error")
    
    with patch('llm_consortium.DatabaseConnection.get_connection') as mock_get_conn:
        mock_get_conn.return_value = mock_db
        with pytest.raises(Exception) as exc_info:
            log_response(mock_response, "test-model")
        assert "Test error" in str(exc_info.value)
