"""FastAPI dependency-injection functions.

Each function in this module is designed to be used with ``Depends(...)`` in
route signatures. They resolve the global `ApplicationContainer` singleton
into the specific manager, repository, or service instance required by the
route handler.

The dependency graph is::

    get_container
    ├── get_config_manager
    ├── get_model_manager
    ├── get_database_repository
    └── get_index_repository
        ├── get_search_service  (model_manager + index_repo + database_repo)
        └── get_item_service    (database_repo + config_manager)
"""

from fastapi import Depends

from app.core.config import ConfigManager
from app.repositories.database_repository import DatabaseRepository
from app.repositories.index_repository import IndexRepository

from ..core.models import ApplicationContainer, ModelManager, container
from ..services.search_service import SearchService
from ..services.item_service import ItemService
from ..services.logging_service import LoggingService, AuditLogger


def get_container() -> ApplicationContainer:
    """Get the application container."""
    return container


def get_config_manager(app_container: ApplicationContainer=Depends(get_container)) -> ConfigManager:
    """Get the configuration manager."""
    return app_container.config_manager

    
def get_model_manager(app_container: ApplicationContainer=Depends(get_container)) -> ModelManager:
    """Get the model manager."""
    return app_container.model_manager


def get_database_repository(app_container: ApplicationContainer=Depends(get_container)) -> DatabaseRepository:
    """Get the database repository."""
    return app_container.database_repository


def get_index_repository(app_container: ApplicationContainer=Depends(get_container)) -> IndexRepository:
    """Get the index repository."""
    return app_container.index_repository


def get_search_service(
    model_manager: ModelManager=Depends(get_model_manager),
    index_repo: IndexRepository=Depends(get_index_repository),
    database_repo: DatabaseRepository=Depends(get_database_repository),
) -> SearchService:
    """Get the search service."""
    return SearchService(model_manager, index_repo, database_repo)


def get_item_service(
    database_repo: DatabaseRepository=Depends(get_database_repository),
    config_manager: ConfigManager=Depends(get_config_manager),
) -> ItemService:
    """Get the item service."""
    return ItemService(database_repo, config_manager)


def get_logging_service() -> LoggingService:
    """Get the logging service."""
    # This will be initialized properly in the main app
    # For now, we'll create a basic instance
    # In production, this should come from the app state
    from fastapi import HTTPException

    raise HTTPException(status_code=500, detail="Logging service not initialized")
