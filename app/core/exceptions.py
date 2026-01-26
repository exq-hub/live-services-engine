"""Custom exceptions for the LSE application."""

from typing import Any, Dict, Optional


class LSEException(Exception):
    """Base exception for all LSE-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(LSEException):
    """Raised when there's a configuration-related error."""

    pass


class ModelLoadError(LSEException):
    """Raised when a model fails to load."""

    pass


class SearchError(LSEException):
    """Raised when a search operation fails."""

    pass


class IndexError(LSEException):
    """Raised when there's an issue with search indices."""

    pass


class MetadataError(LSEException):
    """Raised when there's an issue with metadata operations."""

    pass

class DatabaseError(LSEException):
    """Raised when there's an issue with database operations."""

    pass


class ValidationError(LSEException):
    """Raised when data validation fails."""

    pass


class ServiceUnavailableError(LSEException):
    """Raised when a service is temporarily unavailable."""

    pass
