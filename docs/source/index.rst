.. Live Service Engine documentation master file, created by
   sphinx-quickstart on Wed Aug 27 22:11:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Live Service Engine Documentation
==================================

The Live Service Engine (LSE) handles execution and logging of search requests within the Exquisitor multimedia search system. It provides a FastAPI-based REST API for various search strategies and multimedia content retrieval.

Features
--------

* **Multi-modal Search**: Support for CLIP-based visual search, caption search, and RF (relevance feedback) search
* **FastAPI Integration**: RESTful API endpoints for search operations
* **Modular Architecture**: Clean separation between API, services, repositories, and search strategies
* **Logging and Analytics**: Comprehensive request logging and performance monitoring
* **Extensible Design**: Plugin-based search strategy system

Quick Start
-----------

To get started with the Live Service Engine:

1. Install dependencies: ``uv sync``
2. Run the application: ``uv run uvicorn main:app --reload``
3. Access the API documentation at ``http://localhost:8000/docs``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   development
   api
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

