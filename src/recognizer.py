"""
F2/F3 - SKU Recognition & Open-Set Rejection Module
=====================================================

This module provides:
1. FeatureExtractor: Extracts L2-normalized deep visual embeddings from product crops
   using a pre-trained CNN (ResNet-50 by default).
2. SKUMatcher: Queries a gallery feature index using cosine similarity and handles
   open-set rejection via similarity thresholds.

Conforms to enterprise SQA standards with explicit type hinting, boundary checks,
and informative docstrings.
"""

from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class FeatureExtractor:
    """Extracts compact feature embeddings from product image crops.

    Uses a pre-trained ResNet-50 backbone stripped of its classification head,
    producing L2-normalized embeddings suitable for cosine distance retrieval.

    Attributes:
        device (torch.device): Compute device ('cpu' or 'cuda').
        model (nn.Module): Truncated backbone network.
        transform (transforms.Compose): Image preprocessing pipeline.
        embedding_dim (int): Dimensionality of the output feature vector (2048 for ResNet50).
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: Optional[str] = None,
        weights: str = "DEFAULT",
    ) -> None:
        """Initializes the feature extractor backbone and preprocessing transforms.

        Args:
            backbone: Architecture name ('resnet50').
            device: Target execution device ('cpu', 'cuda', etc.). If None, auto-detects.
            weights: Pretrained weights specification (defaults to PyTorch standard DEFAULT).

        Raises:
            ValueError: If an unsupported backbone architecture is specified.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if backbone == "resnet50":
            # Load ResNet50 with standard torchvision weights
            base_model = models.resnet50(weights=weights)
            # Remove the final classification layer (fc) while retaining the avgpool
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.embedding_dim: int = 2048
        else:
            raise ValueError(f"Unsupported backbone: '{backbone}'. Supported: ['resnet50']")

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization and preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_single(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extracts an L2-normalized embedding for a single image crop.

        Args:
            image: PIL Image or NumPy array (RGB/BGR/Grayscale).

        Returns:
            1D float32 NumPy array of shape (embedding_dim,) normalized to unit length.

        Raises:
            TypeError: If input image is not PIL.Image or np.ndarray.
            ValueError: If input image is empty or invalid.
        """
        batch_embedding = self.extract_batch([image])
        return batch_embedding[0]

    @torch.no_grad()
    def extract_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        batch_size: int = 64
    ) -> np.ndarray:
        """Extracts L2-normalized embeddings for a list of image crops in mini-batches.

        Args:
            images: List of images as PIL Images or NumPy arrays.
            batch_size: Mini-batch size for GPU/CPU evaluation.

        Returns:
            2D float32 NumPy array of shape (N, embedding_dim), each row normalized to unit length.

        Raises:
            ValueError: If the images list is empty.
        """
        if not images:
            raise ValueError("Input images list cannot be empty.")

        tensors = []
        for idx, img in enumerate(images):
            if isinstance(img, np.ndarray):
                assert img.size > 0, f"Image at index {idx} is empty array."
                # Convert OpenCV BGR to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    pil_img = Image.fromarray(img[..., ::-1])
                elif len(img.shape) == 2:
                    pil_img = Image.fromarray(img).convert("RGB")
                else:
                    pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Element at index {idx} has unsupported type {type(img)}.")

            tensors.append(self.transform(pil_img))

        all_embeddings: List[np.ndarray] = []
        num_samples = len(tensors)

        for i in range(0, num_samples, batch_size):
            batch_tensors = torch.stack(tensors[i:i + batch_size]).to(self.device)
            # Forward pass: shape (B, 2048, 1, 1) -> flatten to (B, 2048)
            feats = self.model(batch_tensors).flatten(start_dim=1)
            # L2 normalization across feature dimension
            feats = nn.functional.normalize(feats, p=2, dim=1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))

        result = np.vstack(all_embeddings)
        assert result.shape == (num_samples, self.embedding_dim), (
            f"Expected shape ({num_samples}, {self.embedding_dim}), got {result.shape}"
        )
        return result


class SKUMatcher:
    """Matches query crop embeddings against a gallery with open-set rejection.

    Implements dual-threshold open-set identification:
    - similarity >= match_threshold: CONFIRMED match to top SKU.
    - reject_threshold <= similarity < match_threshold: LOW_CONFIDENCE match.
    - similarity < reject_threshold: REJECTED as UNKNOWN (open-set facing).

    Attributes:
        match_threshold (float): Threshold above which a match is confident.
        reject_threshold (float): Threshold below which a crop is declared unknown.
    """

    def __init__(
        self,
        match_threshold: float = 0.65,
        reject_threshold: float = 0.40
    ) -> None:
        """Initializes the matcher with classification and rejection thresholds.

        Args:
            match_threshold: Minimum cosine similarity for a solid match [0, 1].
            reject_threshold: Lower bound cosine similarity; below this is 'UNKNOWN'.

        Raises:
            ValueError: If thresholds are invalid or reject_threshold >= match_threshold.
        """
        if not (0.0 <= reject_threshold < match_threshold <= 1.0):
            raise ValueError(
                f"Threshold constraints violated. Must satisfy: "
                f"0.0 <= reject ({reject_threshold}) < match ({match_threshold}) <= 1.0"
            )
        self.match_threshold = match_threshold
        self.reject_threshold = reject_threshold

    def match(
        self,
        query_embedding: np.ndarray,
        gallery_index: Any,
        top_k: int = 1
    ) -> Dict[str, Any]:
        """Matches a single query embedding against the gallery index.

        Args:
            query_embedding: 1D array of shape (embedding_dim,) or 2D array (1, embedding_dim).
            gallery_index: Instance of GalleryIndex providing query(embedding, top_k).
            top_k: Number of nearest gallery neighbors to retrieve.

        Returns:
            Dictionary containing:
                - 'sku_id' (str): Best SKU identifier or 'UNKNOWN'.
                - 'confidence' (float): Cosine similarity score [0.0, 1.0].
                - 'status' (str): 'CONFIRMED', 'LOW_CONFIDENCE', or 'UNKNOWN'.
                - 'top_candidates' (List[Dict]): Detailed top_k retrieved records.

        Raises:
            ValueError: If query embedding dimensions are malformed.
        """
        assert query_embedding is not None, "Query embedding cannot be None."
        if query_embedding.ndim == 1:
            q = query_embedding.reshape(1, -1)
        elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            q = query_embedding
        else:
            raise ValueError(f"Expected shape (D,) or (1, D), got {query_embedding.shape}")

        candidates = gallery_index.search(q, top_k=top_k)[0]
        if not candidates:
            return {
                "sku_id": "UNKNOWN",
                "confidence": 0.0,
                "status": "UNKNOWN",
                "top_candidates": []
            }

        top_match = candidates[0]
        best_sku = top_match["sku_id"]
        similarity = float(top_match["score"])

        # Open-set classification logic
        if similarity >= self.match_threshold:
            status = "CONFIRMED"
            sku_id = best_sku
        elif similarity >= self.reject_threshold:
            status = "LOW_CONFIDENCE"
            sku_id = best_sku
        else:
            status = "UNKNOWN"
            sku_id = "UNKNOWN"

        return {
            "sku_id": sku_id,
            "confidence": similarity,
            "status": status,
            "top_candidates": candidates
        }
