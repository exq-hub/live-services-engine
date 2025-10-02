Installation
============

Requirements
------------

* Python 3.13 or higher
* UV package manager (recommended) or pip

Setup
-----

1. **Clone the repository**:

   .. code-block:: bash

      git clone <repository-url>
      cd lse

2. **Install dependencies using UV** (recommended):

   .. code-block:: bash

      uv sync

   Or using pip:

   .. code-block:: bash

      pip install -e .

3. **Run the application**:

   .. code-block:: bash

      uv run uvicorn main:app --reload

   The API will be available at http://localhost:8000

4. **Access API documentation**:

   * Interactive docs: http://localhost:8000/docs
   * ReDoc: http://localhost:8000/redoc

Development Setup
-----------------

For development with additional tools:

.. code-block:: bash

   uv sync --group dev

This installs additional development dependencies including:

* **ruff**: Code linting and formatting

Configuration
-------------

The application can be configured through environment variables or configuration files. See the :doc:`api` section for more details on configuration options.