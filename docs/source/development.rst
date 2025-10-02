Development Guide
=================

This guide covers development practices and contribution guidelines for the Live Service Engine.

Development Setup
-----------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone <repository-url>
      cd lse

2. **Install development dependencies**:

   .. code-block:: bash

      uv sync --group dev

3. **Run the application in development mode**:

   .. code-block:: bash

      uv run uvicorn main:app --reload

Code Quality
------------

The project uses several tools to maintain code quality:

**Ruff**: For linting and formatting

.. code-block:: bash

   uv run ruff check .
   uv run ruff format .

Architecture Overview
---------------------

The Live Service Engine follows a clean architecture pattern with the following layers:

* **API Layer** (``app/api/``): FastAPI routes and request/response handling
* **Service Layer** (``app/services/``): Business logic and orchestration  
* **Repository Layer** (``app/repositories/``): Data access and persistence
* **Strategy Layer** (``app/strategies/``): Search algorithm implementations
* **Core Layer** (``app/core/``): Configuration, models, and exceptions

Key Components
--------------

Search Strategies
~~~~~~~~~~~~~~~~~

The system supports multiple search strategies through a plugin architecture:

* **CLIP Search**: Visual similarity search using CLIP embeddings
* **Caption Search**: Text-based search through video captions
* **RF Search**: Relevance feedback search with positive/negative examples
* **Aggregate Search**: Combines multiple search strategies

Services
~~~~~~~~

* **Search Service**: Orchestrates search operations across strategies
* **Item Service**: Manages item metadata and retrieval
* **Logging Service**: Handles request logging and analytics

Testing
-------

Run tests using:

.. code-block:: bash

   uv run pytest

Contributing
------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

Please ensure your code follows the existing style and includes appropriate tests.