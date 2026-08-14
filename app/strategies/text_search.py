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


"""Text-embedding text-to-item search strategy.

Encodes a natural-language text query with a sentence-transformers model
(via `TextModelManager`) instead of open_clip, then retrieves the *n*
most similar media items -- e.g. searching transcripts by meaning rather
than an image-text CLIP space. Everything but the actual encoding call
lives in `EmbeddingSearchStrategy` (shared with `CLIPSearchStrategy`):
index resolution, skip-set construction, expanding search, and ID mapping.
"""

import asyncio
from functools import partial
from typing import ClassVar

import numpy as np

from .embedding_search import EmbeddingSearchStrategy


class TextEmbeddingSearchStrategy(EmbeddingSearchStrategy):
    """Search strategy using sentence-transformers text embeddings."""

    embedding_type: ClassVar[str] = "Text"

    def get_strategy_name(self) -> str:
        return "Text Search"

    def _sync_encode_text(self, model_name: str, text: str) -> np.ndarray:
        "Synchronous text encoding function to be run in a thread pool."
        encoder = self.model_manager.get_text_encoder(model_name)
        return encoder.encode([text], normalize_embeddings=True)

    async def _encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Asynchronously encode text by running the synchronous encoding function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._sync_encode_text, model_name, text)
        )
