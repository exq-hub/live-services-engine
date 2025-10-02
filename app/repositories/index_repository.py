"""Repository for managing search indices."""

import faiss
import zarr
import compress_pickle as cp
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from ..core.exceptions import IndexError


class IndexRepository:
    """Repository for managing FAISS indices and embeddings."""
    
    def __init__(self):
        self._clip_indices: Dict[str, faiss.Index] = {}
        self._caption_indices: Dict[str, faiss.Index] = {}
        self._pca_models: Dict[str, Dict] = {}
        self._embeddings_zarr: Dict[str, str] = {}
    
    def load_clip_index(self, collection: str, index_path: str) -> faiss.Index:
        """Load CLIP FAISS index for a collection."""
        try:
            if collection in self._clip_indices:
                return self._clip_indices[collection]
            
            index_file = Path(index_path)
            if not index_file.exists():
                raise IndexError(f"CLIP index file not found: {index_path}")
            
            index = faiss.read_index(str(index_file))
            self._clip_indices[collection] = index
            return index
        
        except Exception as e:
            raise IndexError(f"Failed to load CLIP index from {index_path}: {e}")
    
    def load_caption_index(self, collection: str, index_path: str) -> faiss.Index:
        """Load caption FAISS index for a collection."""
        try:
            if collection in self._caption_indices:
                return self._caption_indices[collection]
            
            index_file = Path(index_path)
            if not index_file.exists():
                raise IndexError(f"Caption index file not found: {index_path}")
            
            index = faiss.read_index(str(index_file))
            self._caption_indices[collection] = index
            return index
        
        except Exception as e:
            raise IndexError(f"Failed to load caption index from {index_path}: {e}")
    
    def load_pca_data(self, collection: str, pca_model_path: str, 
                      scaler_path: str, embeddings_path: str) -> Dict:
        """Load PCA model, scaler, and embeddings."""
        try:
            if collection in self._pca_models:
                return self._pca_models[collection]
            
            # Load PCA model
            pca_file = Path(pca_model_path)
            if not pca_file.exists():
                raise IndexError(f"PCA model file not found: {pca_model_path}")
            
            with open(pca_file, "rb") as f:
                pca_model = cp.load(f, compression="gzip")
            
            # Load scaler
            scaler_file = Path(scaler_path)
            if not scaler_file.exists():
                raise IndexError(f"Scaler file not found: {scaler_path}")
            
            with open(scaler_file, "rb") as f:
                scaler = cp.load(f, compression="gzip")
            
            # Validate embeddings file
            embeddings_file = Path(embeddings_path)
            if not embeddings_file.exists():
                raise IndexError(f"PCA embeddings file not found: {embeddings_path}")
            
            pca_data = {
                "model": pca_model,
                "scaler": scaler,
                "embeddings_f": str(embeddings_file),
            }
            
            self._pca_models[collection] = pca_data
            return pca_data
        
        except Exception as e:
            raise IndexError(f"Failed to load PCA data for {collection}: {e}")
    
    def set_embeddings_zarr_path(self, collection: str, embeddings_path: str):
        """Set the path to embeddings zarr file for a collection."""
        embeddings_file = Path(embeddings_path)
        if not embeddings_file.exists():
            raise IndexError(f"Embeddings file not found: {embeddings_path}")
        
        self._embeddings_zarr[collection] = str(embeddings_file)
    
    def get_clip_index(self, collection: str) -> Optional[faiss.Index]:
        """Get CLIP index for a collection."""
        return self._clip_indices.get(collection)
    
    def get_caption_index(self, collection: str) -> Optional[faiss.Index]:
        """Get caption index for a collection."""
        return self._caption_indices.get(collection)
    
    def get_pca_data(self, collection: str) -> Optional[Dict]:
        """Get PCA data for a collection."""
        return self._pca_models.get(collection)
    
    def get_embeddings_zarr_path(self, collection: str) -> Optional[str]:
        """Get embeddings zarr path for a collection."""
        return self._embeddings_zarr.get(collection)
    
    def search_clip(self, collection: str, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search CLIP index."""
        index = self.get_clip_index(collection)
        if index is None:
            raise IndexError(f"CLIP index not loaded for collection: {collection}")
        
        try:
            distances, indices = index.search(query_vector, k)
            return distances, indices
        except Exception as e:
            raise IndexError(f"CLIP search failed for collection {collection}: {e}")
    
    def search_caption(self, collection: str, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search caption index."""
        index = self.get_caption_index(collection)
        if index is None:
            raise IndexError(f"Caption index not loaded for collection: {collection}")
        
        try:
            distances, indices = index.search(query_vector, k)
            return distances, indices
        except Exception as e:
            raise IndexError(f"Caption search failed for collection {collection}: {e}")
    
    def get_embeddings_array(self, collection: str) -> zarr.Array:
        """Get zarr embeddings array for a collection."""
        pca_data = self.get_pca_data(collection)
        if pca_data:
            store = zarr.storage.ZipStore(pca_data["embeddings_f"], mode="r")
        else:
            zarr_path = self.get_embeddings_zarr_path(collection)
            if zarr_path is None:
                raise IndexError(f"No embeddings configured for collection: {collection}")
            store = zarr.storage.ZipStore(zarr_path, mode="r")
        
        try:
            emb_arr = zarr.open(store, mode="r")["embeddings"]
            return emb_arr
        except Exception as e:
            raise IndexError(f"Failed to open embeddings array for collection {collection}: {e}")
    
    def clear_cache(self, collection: Optional[str] = None):
        """Clear cached indices for a collection or all collections."""
        if collection:
            self._clip_indices.pop(collection, None)
            self._caption_indices.pop(collection, None)
            self._pca_models.pop(collection, None)
            self._embeddings_zarr.pop(collection, None)
        else:
            self._clip_indices.clear()
            self._caption_indices.clear()
            self._pca_models.clear()
            self._embeddings_zarr.clear()