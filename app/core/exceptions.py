"""Custom exception hierarchy for the LSE application.

All application-specific exceptions inherit from `LSEException`, which
carries a human-readable ``message`` and an optional ``details`` dictionary
for structured context. The global exception handler in ``main.py`` catches
any `LSEException` and converts it to an HTTP 400 response while also
recording it in the audit log.

Hierarchy::

    LSEException
    ├── ConfigurationError   -- invalid or missing configuration
    ├── ModelLoadError        -- ML model loading/device failures
    ├── SearchError           -- search strategy execution failures
    ├── IndexError            -- vector index issues (FAISS/Zarr)
    ├── MetadataError         -- metadata retrieval issues
    ├── DatabaseError         -- SQLite query / connection failures
    ├── ValidationError       -- request data validation failures
    └── ServiceUnavailableError -- transient unavailability
"""

from typing import Any, Dict, Optional


class LSEException(Exception):
    """Base exception for all LSE-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message: str = message
        """Human-readable error description."""

        self.details: Dict[str, Any] = details or {}
        """Structured context dict included in HTTP error responses and audit logs."""


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
