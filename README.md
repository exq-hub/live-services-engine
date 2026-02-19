# Live Services Engine (LSE)

The Live Services Engine is the backend for [Exquisitor](https://exquisitor.org/), a multimedia search and exploration system. It provides text-to-image similarity search using CLIP, relevance feedback using SVMs, and faceted metadata filtering across large media collections.

## Features

- **CLIP search** -- encode a text query and find visually similar images using CLIP embeddings (ViT-SO400M-14-SigLIP-384)
- **Relevance feedback** -- refine results by marking positive/negative examples; an SVM learns a decision boundary over the embedding space
- **Faceted filtering** -- recursive filter expressions (AND/OR trees with negation) over structured metadata
- **Multi-collection support** -- each collection has its own database, indices, and media URLs, configured independently
- **Audit logging** -- all search requests, item views, and client events are logged in MessagePack format
- **Auto device selection** -- automatically uses CUDA, MPS, or CPU depending on availability

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/exq-hub/live-services-engine
cd live-services-engine
uv sync
```

## Configuration

Copy the template and edit it:

```bash
cp data/config.ini.template data/config.ini
```

A minimal configuration looks like:

```ini
[DEFAULT]
ModelDevice = auto

[SERVER]
Host = 127.0.0.1
Port = 8000
Reload = True

[LOGGING]
Level = INFO

# Zarr collection — EmbeddingsFile is used directly as the brute-force index:
[my_collection]
Enabled = True
IndexType = zarr
EmbeddingsFile = ./data/my_collection/embeddings.zarr.zip
DatabaseFile = ./data/my_collection/my_collection.db
ThumbnailMediaURL = https://localhost:5001/my_collection
OriginalMediaURL = https://localhost:5001/my_collection

# FAISS collection — separate ANN index and raw embeddings required:
# [my_faiss_collection]
# Enabled = True
# IndexType = faiss
# CLIPIndexFile = ./data/my_faiss_collection/clip_index.faiss
# EmbeddingsFile = ./data/my_faiss_collection/embeddings.zarr.zip
# DatabaseFile = ./data/my_faiss_collection/my_collection.db
# ThumbnailMediaURL = https://localhost:5001/my_faiss_collection
# OriginalMediaURL = https://localhost:5001/my_faiss_collection
```

Each `[section]` (other than `DEFAULT`, `SERVER`, and `LOGGING`) defines a collection.

## Usage

```bash
# Using uv
uv run python main.py

# Using invoke
uv run invoke run --host=0.0.0.0 --port=8000

# Stop the server
uv run invoke stop
```

Once running, the interactive API docs are available at `http://localhost:8000/docs`.

## API

All endpoints are mounted under `/exq/`. We may introduce a semantic versioning code into the API endpoint before release.

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/exq/search/clip` | Text-to-image search using CLIP |
| POST | `/exq/search/rf` | Relevance feedback search |
| POST | `/exq/search/faceted` | Filter-only search |

### Items

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/exq/item/base` | Basic item info (URI, thumbnails) |
| POST | `/exq/item/details` | Detailed metadata for selected filters |
| POST | `/exq/item/related` | Related items by group |
| POST | `/exq/item/excluded` | Check if item belongs to an excluded group |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/exq/init/{session}` | Initialize a session, returns available collections |
| POST | `/exq/info/totalItems` | Total item count for a collection |
| GET | `/exq/info/filters/{session}/{collection}` | Available filter definitions |
| POST | `/exq/log/clientEvent` | Batch-log client UI events |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | Service info |

## Project Structure

```
app/
  api/routes/        # FastAPI route handlers (search, items, admin)
  core/              # Config, model management, index types, exceptions
  services/          # Business logic (search dispatch, item retrieval, logging)
  repositories/      # Data access (SQLite metadata, FAISS/Zarr vector indices)
  strategies/        # Search implementations (CLIP, relevance feedback, faceted)
  schemas/           # Pydantic request/response models
```

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format .
```

## Citation

If you find this project or any of Exquisitor's subcomponents useful, please cite:

```bibtex
@inproceedings{sharma2025can,
  title={Can relevance feedback, conversational search and foundation models work together for interactive video search and exploration?},
  author={Sharma, Ujjwal and Khan, Omar Shahbaz and Rudinac, Stevan and J{\'o}nsson, Bj{\"o}rn {\TH}{\'o}r},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3740--3749},
  year={2025}
}
```

## License

GNU Affero General Public License v3.0 or later.
# Live Services Engine (LSE)

The Live Services Engine is the backend for [Exquisitor](https://exquisitor.org/), a multimedia search system. It provides text-to-image similarity search using CLIP, relevance feedback using SVMs, and faceted metadata filtering across large media collections.

## Features

- **CLIP search** -- encode a text query and find visually similar images using CLIP embeddings (ViT-SO400M-14-SigLIP-384)
- **Relevance feedback** -- refine results by marking positive/negative examples; an SVM learns a decision boundary over the embedding space
- **Faceted filtering** -- recursive filter expressions (AND/OR trees with negation) over structured metadata
- **Multi-collection support** -- each collection has its own database, indices, and media URLs, configured independently
- **Audit logging** -- all search requests, item views, and client events are logged in MessagePack format
- **Auto device selection** -- automatically uses CUDA, MPS, or CPU depending on availability

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone <repository-url>
cd live-services-engine
uv sync
```

## Configuration

Copy the template and edit it:

```bash
cp data/config.ini.template data/config.ini
```

A minimal configuration looks like:

```ini
[DEFAULT]
ModelDevice = auto

[SERVER]
Host = 127.0.0.1
Port = 8000
Reload = True

[LOGGING]
Level = INFO

# Zarr collection — EmbeddingsFile is used directly as the brute-force index:
[my_collection]
Enabled = True
IndexType = zarr
EmbeddingsFile = ./data/my_collection/embeddings.zarr.zip
DatabaseFile = ./data/my_collection/my_collection.db
ThumbnailMediaURL = https://localhost:5001/my_collection
OriginalMediaURL = https://localhost:5001/my_collection

# FAISS collection — separate ANN index and raw embeddings required:
# [my_faiss_collection]
# Enabled = True
# IndexType = faiss
# CLIPIndexFile = ./data/my_faiss_collection/clip_index.faiss
# EmbeddingsFile = ./data/my_faiss_collection/embeddings.zarr.zip
# DatabaseFile = ./data/my_faiss_collection/my_collection.db
# ThumbnailMediaURL = https://localhost:5001/my_faiss_collection
# OriginalMediaURL = https://localhost:5001/my_faiss_collection
```

Each `[section]` (other than `DEFAULT`, `SERVER`, and `LOGGING`) defines a collection.

## Usage

```bash
# Using uv
uv run python main.py

# Using invoke
uv run invoke run --host=0.0.0.0 --port=8000

# Stop the server
uv run invoke stop
```

Once running, the interactive API docs are available at `http://localhost:8000/docs`.

## API

All endpoints are mounted under `/exq/`.

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/exq/search/clip` | Text-to-image search using CLIP |
| POST | `/exq/search/rf` | Relevance feedback search |
| POST | `/exq/search/faceted` | Filter-only search |

### Items

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/exq/item/base` | Basic item info (URI, thumbnails) |
| POST | `/exq/item/details` | Detailed metadata for selected filters |
| POST | `/exq/item/related` | Related items by group |
| POST | `/exq/item/excluded` | Check if item belongs to an excluded group |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/exq/init/{session}` | Initialize a session, returns available collections |
| POST | `/exq/info/totalItems` | Total item count for a collection |
| GET | `/exq/info/filters/{session}/{collection}` | Available filter definitions |
| POST | `/exq/log/clientEvent` | Batch-log client UI events |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | Service info |

## Project Structure

```
app/
  api/routes/        # FastAPI route handlers (search, items, admin)
  core/              # Config, model management, index types, exceptions
  services/          # Business logic (search dispatch, item retrieval, logging)
  repositories/      # Data access (SQLite metadata, FAISS/Zarr vector indices)
  strategies/        # Search implementations (CLIP, relevance feedback, faceted)
  schemas/           # Pydantic request/response models
```

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format .
```

## Citation

If you find this project or any of Exquisitor's subcomponents useful, please cite:

```bibtex
@inproceedings{sharma2025can,
  title={Can relevance feedback, conversational search and foundation models work together for interactive video search and exploration?},
  author={Sharma, Ujjwal and Khan, Omar Shahbaz and Rudinac, Stevan and J{\'o}nsson, Bj{\"o}rn {\TH}{\'o}r},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3740--3749},
  year={2025}
}
```

## License

GNU Affero General Public License v3.0 or later.
