"""FastAPI dependencies for dependency injection."""

from fastapi import Depends

from ..core.models import container
from ..services.search_service import SearchService
from ..services.item_service import ItemService
from ..services.logging_service import LoggingService, AuditLogger


def get_container():
    """Get the application container."""
    return container


def get_config_manager(app_container=Depends(get_container)):
    """Get the configuration manager."""
    return app_container.config_manager


def get_model_manager(app_container=Depends(get_container)):
    """Get the model manager."""
    return app_container.model_manager


def get_metadata_repository(app_container=Depends(get_container)):
    """Get the metadata repository."""
    return app_container.metadata_repository


def get_index_repository(app_container=Depends(get_container)):
    """Get the index repository."""
    return app_container.index_repository


def get_search_service(
    model_manager=Depends(get_model_manager),
    index_repo=Depends(get_index_repository),
    metadata_repo=Depends(get_metadata_repository),
) -> SearchService:
    """Get the search service."""
    return SearchService(model_manager, index_repo, metadata_repo)


def get_item_service(
    metadata_repo=Depends(get_metadata_repository),
    config_manager=Depends(get_config_manager),
) -> ItemService:
    """Get the item service."""
    return ItemService(metadata_repo, config_manager)


def get_logging_service() -> LoggingService:
    """Get the logging service."""
    # This will be initialized properly in the main app
    # For now, we'll create a basic instance
    # In production, this should come from the app state
    from fastapi import HTTPException

    raise HTTPException(status_code=500, detail="Logging service not initialized")
