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
cp data/config.toml.template data/config.toml
```

A minimal configuration looks like:

```toml
[default]
model_device = "auto"

[server]
host = "127.0.0.1"
port = 8000
reload = true

[logging]
level = "INFO"

# A collection declares one or more indexes, each independently picking
# its own index_type ("zarr" uses the embeddings file directly as a
# brute-force index; "faiss" points index_file at a separate ANN
# structure) and embedding_type ("CLIP" or "Text") -- a single
# collection can freely mix both.
[[collections]]
name = "my_collection"
enabled = true
database_file = "./data/my_collection/my_collection.db"
thumbnail_media_url = "https://localhost:5001/my_collection"
original_media_url = "https://localhost:5001/my_collection"

  [[collections.indexes]]
  name = "CLIP"
  index_type = "zarr"
  embeddings_file = "./data/my_collection/embeddings.zarr.zip"

# A collection with more than one index must mark exactly one default = true.
# [[collections.indexes]]
# name = "Transcripts"
# index_type = "faiss"
# index_file = "./data/my_collection/transcripts.faiss"
# embeddings_file = "./data/my_collection/transcript_embeddings.zip"
# embedding_type = "Text"
# model_name = "your-sentence-transformers-model"
# source_type = "Text"
```

Each `[[collections]]` table defines a collection, with one or more nested `[[collections.indexes]]` tables for its embeddings/index configuration. See `data/config.toml.template` for the full set of per-index options (`model_name`, `source_type`, `tagset`, `preload_all_indexes`, etc.).

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
| POST | `/exq/search/text` | Text search using sentence-transformers (e.g. transcripts) |
| POST | `/exq/search/rf` | Relevance feedback search |
| POST | `/exq/search/faceted` | Filter-only search |

`/clip`, `/text`, and `/rf` all accept an optional `index_name` to target a specific index within a collection (of the matching `embedding_type` for `/clip`/`/text`); omitted, each falls back to the collection's default index for that family.

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
| GET | `/exq/info/filters/values/{session}/{collection}/{tagtypeId}/{tagsetId}` | Possible values for a specific filter |
| GET | `/exq/info/indexes/{session}/{collection}` | Configured indexes for a collection (name, index_type, embedding_type, source_type, default) |
| POST | `/exq/log/addModel` | Audit-log a relevance-feedback model being added |
| POST | `/exq/log/removeModel` | Audit-log a relevance-feedback model being removed |
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