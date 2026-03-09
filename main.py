# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Application entry point for the Live Services Engine (LSE).

This module configures and launches the FastAPI application that serves as the
backend for Exquisitor's multimedia search system. It handles:

- **Application lifecycle**: Startup initialization of ML models, databases, and
  indices; graceful shutdown of logging services.
- **Middleware**: CORS configuration for cross-origin access.
- **Routing**: Mounts the search, item, and admin route modules under the `/exq/` prefix.
- **Error handling**: Global exception handler that catches `LSEException` subclasses,
  logs them to the audit system, and returns structured HTTP error responses.
- **Health monitoring**: `/health` endpoint that validates collection availability,
  database connectivity, and index readiness.

Usage::

    # Direct execution
    python main.py

    # Via invoke tasks
    invoke run --host=0.0.0.0 --port=8000

    # Via uv
    uv run python main.py
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.models import container
from app.core.exceptions import LSEException, ConfigurationError
from app.api.routes import search, items, admin
from app.repositories.database_repository import DatabaseRepository
from app.services.logging_service import AuditLogger, LoggingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""Module-level logger for startup, shutdown, and health-check messages."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        logger.info("Initializing LSE application...")

        # Initialize the application container
        container.initialize()

        # Initialize logging service
        config = container.config_manager.config
        # Use the first collection's log directory, or default
        if config.collections and config.collection_configs:
            first_collection = config.collections[0]
            log_dir = (
                config.collection_configs[first_collection].log_directory or "./logs"
            )
        else:
            log_dir = "./logs"

        audit_logger = AuditLogger(f"{log_dir}/audit.log")
        logging_service = LoggingService(audit_logger)
        await logging_service.start()

        # Store in app state for dependency injection
        app.state.logging_service = logging_service
        app.state.container = container

        logger.info("LSE application initialized successfully")

    except ConfigurationError as e:
        logger.error(f"Configuration error during startup: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        raise

    yield

    # Shutdown
    try:
        logger.info("Shutting down LSE application...")
        if hasattr(app.state, "logging_service"):
            await app.state.logging_service.stop()
        logger.info("LSE application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="Live Services Engine",
    description="The Live Service Engine handles execution and logging of search requests within Exquisitor.",
    version="0.2.0",
    lifespan=lifespan,
)
"""FastAPI application instance for the Live Services Engine."""

origins: list[str] = ["*"]
"""Allowed CORS origins (defaults to all origins for development)."""
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router)
app.include_router(items.router)
app.include_router(admin.router)


# Global exception handler for LSE exceptions
@app.exception_handler(LSEException)
async def lse_exception_handler(request, exc: LSEException):
    """Handle LSE-specific exceptions."""
    logger.error(
        f"LSE Exception: {exc.message}",
        extra={
            "exception_type": type(exc).__name__,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    # Log to audit system if available
    if hasattr(app.state, "logging_service"):
        try:
            await app.state.logging_service.log_error(
                exc,
                session="system",
                context={"path": request.url.path, "method": request.method},
            )
        except Exception:
            pass  # Don't fail the request if logging fails

    return HTTPException(status_code=400, detail=exc.message)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health checks
        config = container.config_manager.config
        collections = config.collections

        # Check if at least one collection is available
        if not collections:
            return {"status": "unhealthy", "reason": "No collections configured"}

        # Check if repositories are initialized
        metadata_repo = container.metadata_repository
        index_repo = container.index_repository

        # Quick validation that data is loaded
        for collection in collections[:1]:  # Check first collection
            if isinstance(metadata_repo, DatabaseRepository):
                if not metadata_repo.is_loaded(collection):
                    return {
                        "status": "unhealthy",
                        "reason": f"Database not loaded for {collection}",
                    }

            if not index_repo.get_clip_index(collection):
                return {
                    "status": "unhealthy",
                    "reason": f"CLIP index not loaded for {collection}",
                }

        return {
            "status": "healthy",
            "version": "0.2.0",
            "collections": collections,
            "models_loaded": {
                "clip": True,
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Live Services Engine",
        "version": "0.2.0",
        "description": "The Live Service Engine handles execution and logging of search requests within Exquisitor.",
        "endpoints": {
            "health": "/health",
            "search": "/exq/search/",
            "items": "/exq/item/",
            "admin": "/exq/",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Get configuration for server settings
    try:
        config = container.config_manager.config
        uvicorn.run(
            "main:app",
            host=config.host,
            port=config.port,
            reload=config.reload,
            log_level=config.log_level.lower(),
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)
