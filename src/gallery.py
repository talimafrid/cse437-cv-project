"""
F6 & Gallery Indexing Module
============================

Manages reference SKU embeddings using Faiss IndexFlatIP (Inner Product)
for fast cosine similarity search over unit-normalized feature vectors.

Supports:
- Building gallery from GroZi training reference images.
- Dynamic addition of new SKUs without retraining (F6 Add-a-SKU).
- Persistence (saving & loading faiss index + metadata).
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
from pathlib import Path
import numpy as np
import faiss
from PIL import Image

from src.recognizer import FeatureExtractor


class GalleryIndex:
    """Manages the vector database for SKU gallery reference images using Faiss.

    Since all feature vectors are L2-normalized upon extraction, the Inner Product (IP)
    metric directly computes cosine similarity:
        similarity = dot(u, v) in [-1.0, 1.0]

    Attributes:
        embedding_dim (int): Vector dimension (e.g. 2048 for ResNet50).
        index (faiss.IndexFlatIP): Underlying Faiss index.
        metadata (List[Dict[str, Any]]): Parallel metadata mapping index IDs to SKU info.
    """

    def __init__(self, embedding_dim: int = 2048) -> None:
        """Initializes an empty Faiss Inner Product index.

        Args:
            embedding_dim: Dimensionality of feature embeddings.
        """
        assert embedding_dim > 0, "Embedding dimension must be strictly positive."
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        """Returns total number of items currently indexed in the gallery."""
        return self.index.ntotal

    def add_sku(
        self,
        sku_id: str,
        image_path_or_crop: Union[str, Path, Image.Image, np.ndarray],
        extractor: FeatureExtractor,
        category: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Dynamically registers a single SKU into the gallery index (F6: Add-a-SKU).

        Does not require re-indexing or retraining.

        Args:
            sku_id: Unique identifier for the product SKU (e.g., 'Food/Tea/1').
            image_path_or_crop: Filepath or image object of the reference product.
            extractor: FeatureExtractor instance to compute the embedding.
            category: Optional taxonomy category (e.g., 'Food/Tea').
            custom_metadata: Extra arbitrary fields to attach to this gallery item.

        Returns:
            The integer index assigned to the newly registered SKU.

        Raises:
            FileNotFoundError: If a provided image path does not exist.
            ValueError: If sku_id is empty or image cannot be processed.
        """
        if not sku_id or not str(sku_id).strip():
            raise ValueError("sku_id must be a non-empty string.")

        if isinstance(image_path_or_crop, (str, Path)):
            path_obj = Path(image_path_or_crop)
            if not path_obj.exists():
                raise FileNotFoundError(f"Reference image not found: {path_obj}")
            img = Image.open(path_obj).convert("RGB")
            rel_path = str(path_obj)
        elif isinstance(image_path_or_crop, (Image.Image, np.ndarray)):
            img = image_path_or_crop
            rel_path = "dynamic_in_memory"
        else:
            raise TypeError(f"Unsupported image type: {type(image_path_or_crop)}")

        # Extract 2048-dim L2-normalized embedding
        embedding = extractor.extract_single(img).reshape(1, -1).astype(np.float32)
        assert embedding.shape == (1, self.embedding_dim), (
            f"Embedding dimension mismatch: {embedding.shape} vs expected (1, {self.embedding_dim})"
        )

        assigned_id = self.index.ntotal
        self.index.add(embedding)

        meta = {
            "index_id": assigned_id,
            "sku_id": str(sku_id),
            "image_path": rel_path,
            "category": category if category else "Uncategorized"
        }
        if custom_metadata:
            meta.update(custom_metadata)

        self.metadata.append(meta)
        assert len(self.metadata) == self.index.ntotal, "Metadata and Faiss index length out of sync!"
        return assigned_id

    def build_from_directory(
        self,
        gallery_dir: Union[str, Path],
        extractor: FeatureExtractor,
        batch_size: int = 64,
        valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    ) -> int:
        """Populates the gallery index by scanning a directory tree of reference images.

        Organizes SKU IDs based on relative directory paths.
        For example, 'Food/Tea/1.jpg' -> sku_id = 'Food/Tea/1'.

        Args:
            gallery_dir: Root directory of reference images (e.g. data/grozi/Training).
            extractor: FeatureExtractor instance.
            batch_size: Batch size for feature extraction.
            valid_extensions: Tuple of file extensions considered valid image files.

        Returns:
            Total count of images successfully added to the gallery.
        """
        root_path = Path(gallery_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Gallery directory does not exist: {root_path}")

        image_paths: List[Path] = []
        for ext in valid_extensions:
            image_paths.extend(root_path.rglob(f"*{ext}"))
            image_paths.extend(root_path.rglob(f"*{ext.upper()}"))

        image_paths = sorted(list(set(image_paths)))
        if not image_paths:
            print(f"Warning: No valid images found in {root_path}")
            return 0

        print(f"Building gallery index from {len(image_paths)} images in {root_path}...")

        batch_images: List[Image.Image] = []
        batch_meta: List[Dict[str, Any]] = []

        for idx, img_p in enumerate(image_paths):
            try:
                img = Image.open(img_p).convert("RGB")
                rel_path = img_p.relative_to(root_path)
                sku_id = str(rel_path.with_suffix("")).replace("\\", "/")
                category = str(rel_path.parent).replace("\\", "/")

                batch_images.append(img)
                batch_meta.append({
                    "sku_id": sku_id,
                    "image_path": str(img_p),
                    "category": category
                })
            except Exception as e:
                print(f"Skipping corrupt image {img_p}: {e}")
                continue

            if len(batch_images) == batch_size or (idx == len(image_paths) - 1 and batch_images):
                embeddings = extractor.extract_batch(batch_images, batch_size=batch_size)
                self.index.add(embeddings.astype(np.float32))

                for m in batch_meta:
                    m["index_id"] = len(self.metadata)
                    self.metadata.append(m)

                batch_images = []
                batch_meta = []
                if (idx + 1) % 500 == 0 or idx == len(image_paths) - 1:
                    print(f"  Indexed {self.index.ntotal}/{len(image_paths)} references...")

        return self.index.ntotal

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Performs nearest-neighbor search for one or multiple query vectors.

        Args:
            query_embeddings: 2D array of shape (N, embedding_dim) normalized to unit length.
            top_k: Number of nearest matches to return per query vector.

        Returns:
            List of size N, where each element is a list of top_k dictionaries:
            [{'sku_id': str, 'score': float, 'category': str, 'image_path': str}, ...]
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]

        assert query_embeddings.ndim == 2, f"Expected 2D query array, got shape {query_embeddings.shape}"
        assert query_embeddings.shape[1] == self.embedding_dim, (
            f"Query feature dimension {query_embeddings.shape[1]} doesn't match index {self.embedding_dim}"
        )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embeddings.astype(np.float32), k)

        results: List[List[Dict[str, Any]]] = []
        for b in range(query_embeddings.shape[0]):
            b_results: List[Dict[str, Any]] = []
            for rank in range(k):
                idx = indices[b][rank]
                if idx < 0 or idx >= len(self.metadata):
                    continue
                score = float(scores[b][rank])
                meta = self.metadata[idx].copy()
                meta["score"] = score
                meta["rank"] = rank + 1
                b_results.append(meta)
            results.append(b_results)

        return results

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persists the Faiss index and metadata JSON to disk.

        Args:
            output_dir: Target directory for index and metadata files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        faiss_path = out / "gallery.index"
        meta_path = out / "gallery_metadata.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Gallery saved successfully: {self.index.ntotal} SKUs -> {out}")

    def load(self, input_dir: Union[str, Path]) -> None:
        """Loads a persisted Faiss index and metadata JSON from disk.

        Args:
            input_dir: Directory containing gallery.index and gallery_metadata.json.

        Raises:
            FileNotFoundError: If index or metadata file is missing.
        """
        inp = Path(input_dir)
        faiss_path = inp / "gallery.index"
        meta_path = inp / "gallery_metadata.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"Missing Faiss index file: {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing gallery metadata: {meta_path}")

        self.index = faiss.read_index(str(faiss_path))
        self.embedding_dim = self.index.d

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        assert len(self.metadata) == self.index.ntotal, (
            f"Loaded metadata size ({len(self.metadata)}) != index total ({self.index.ntotal})"
        )
        print(f"Gallery loaded: {self.index.ntotal} SKUs from {inp}")
