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


"""CLIP text-to-image search strategy.

Encodes a natural-language text query into the CLIP embedding space using
the collection's CLIP text encoder, then retrieves the *n* most similar
media items from the collection's vector index.

Everything but the actual encoding call lives in `EmbeddingSearchStrategy`
(shared with `TextEmbeddingSearchStrategy`): index resolution, skip-set
construction, expanding search, and ID mapping.
"""

import asyncio
import contextlib
from functools import partial
from typing import ClassVar

import torch
import numpy as np

from .embedding_search import EmbeddingSearchStrategy


class CLIPSearchStrategy(EmbeddingSearchStrategy):
    """Search strategy using CLIP text embeddings."""

    embedding_type: ClassVar[str] = "CLIP"

    def get_strategy_name(self) -> str:
        return "CLIP Search"

    def _sync_encode_text(self, model_name: str, text: str) -> np.ndarray:
        "Synchronous text encoding function to be run in a thread pool."
        device = self.model_manager.device
        tokenizer = self.model_manager.get_tokenizer(model_name)
        text_model = self.model_manager.get_text_encoder(model_name)

        with (
            torch.inference_mode(),
            (
                torch.amp.autocast("cuda")
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            ),
        ):
            tokenized_text = tokenizer([text]).to(device)
            text_features = text_model(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.detach().cpu().numpy()

    async def _encode_text(self, model_name: str, text: str) -> np.ndarray:
        """Asynchronously encode text using CLIP by running the synchronous encoding function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._sync_encode_text, model_name, text)
        )
