from __future__ import annotations

# embedding/training/paired_data_loader.py
"""
PyTorch Data Loaders for Cross-Modal Contrastive Training

This module provides data loading utilities for training the cross-modal
alignment component of the CMM embedding architecture.

Features:
- Load paired data from corpus files
- In-batch negative sampling for contrastive learning
- Hard negative mining
- Text augmentation for data efficiency
- Multi-modal batch collation

Usage:
    dataset = CMMPairedDataset("cmm_corpus.jsonl")
    dataloader = create_contrastive_dataloader(dataset, batch_size=32)

    for batch in dataloader:
        loss = model(batch["modality_a"], batch["modality_b"])
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PairedExample:
    """A single paired example for training."""

    pair_id: str
    modality_a_type: str
    modality_a_content: Any
    modality_a_metadata: dict[str, Any]
    modality_b_type: str
    modality_b_content: Any
    modality_b_metadata: dict[str, Any]
    confidence_score: float
    bridging_entities: list[str]
    bridging_text: str | None = None


@dataclass
class ContrastiveBatch:
    """A batch for contrastive learning."""

    # Modality A data
    modality_a_types: list[str]
    modality_a_contents: list[Any]
    modality_a_tensors: torch.Tensor | None = None  # Preprocessed tensors

    # Modality B data
    modality_b_types: list[str]
    modality_b_contents: list[Any]
    modality_b_tensors: torch.Tensor | None = None

    # Labels for contrastive learning (positive pairs are on diagonal)
    labels: torch.Tensor | None = None

    # Additional metadata
    pair_ids: list[str] | None = None
    confidence_scores: torch.Tensor | None = None

    def to(self, device: torch.device) -> ContrastiveBatch:
        """Move tensors to device."""
        return ContrastiveBatch(
            modality_a_types=self.modality_a_types,
            modality_a_contents=self.modality_a_contents,
            modality_a_tensors=self.modality_a_tensors.to(device)
            if self.modality_a_tensors is not None
            else None,
            modality_b_types=self.modality_b_types,
            modality_b_contents=self.modality_b_contents,
            modality_b_tensors=self.modality_b_tensors.to(device)
            if self.modality_b_tensors is not None
            else None,
            labels=self.labels.to(device) if self.labels is not None else None,
            pair_ids=self.pair_ids,
            confidence_scores=self.confidence_scores.to(device)
            if self.confidence_scores is not None
            else None,
        )


# =============================================================================
# Dataset Classes
# =============================================================================


class CMMPairedDataset(Dataset):
    """
    PyTorch Dataset for CMM cross-modal paired data.

    Loads pairs from a JSONL corpus file and provides access for training.
    """

    def __init__(
        self,
        corpus_path: str,
        min_confidence: float = 0.0,
        modality_filter: tuple[str, str] | None = None,
        max_text_length: int = 512,
        augment: bool = False,
    ):
        """
        Args:
            corpus_path: Path to JSONL corpus file
            min_confidence: Minimum confidence score to include pair
            modality_filter: Optional tuple of (modality_a, modality_b) to filter
            max_text_length: Maximum text length in characters
            augment: Whether to apply data augmentation
        """
        self.corpus_path = corpus_path
        self.min_confidence = min_confidence
        self.modality_filter = modality_filter
        self.max_text_length = max_text_length
        self.augment = augment

        self.pairs: list[PairedExample] = []
        self.metadata: dict[str, Any] = {}

        self._load_corpus()

    def _load_corpus(self):
        """Load pairs from corpus file."""
        logger.info(f"Loading corpus from {self.corpus_path}")

        with open(self.corpus_path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)

                # First line may be metadata
                if i == 0 and "_metadata" in data:
                    self.metadata = data["_metadata"]
                    continue

                # Filter by confidence
                if data.get("confidence_score", 0) < self.min_confidence:
                    continue

                # Filter by modality if specified
                if self.modality_filter:
                    mod_a = data["modality_a"]["modality"]
                    mod_b = data["modality_b"]["modality"]
                    if not (
                        (mod_a == self.modality_filter[0] and mod_b == self.modality_filter[1])
                        or (mod_a == self.modality_filter[1] and mod_b == self.modality_filter[0])
                    ):
                        continue

                pair = PairedExample(
                    pair_id=data["pair_id"],
                    modality_a_type=data["modality_a"]["modality"],
                    modality_a_content=data["modality_a"]["content"],
                    modality_a_metadata=data["modality_a"].get("metadata", {}),
                    modality_b_type=data["modality_b"]["modality"],
                    modality_b_content=data["modality_b"]["content"],
                    modality_b_metadata=data["modality_b"].get("metadata", {}),
                    confidence_score=data.get("confidence_score", 1.0),
                    bridging_entities=data.get("bridging_entities", []),
                    bridging_text=data.get("bridging_text"),
                )
                self.pairs.append(pair)

        logger.info(f"Loaded {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairedExample:
        pair = self.pairs[idx]

        if self.augment:
            pair = self._augment_pair(pair)

        return pair

    def _augment_pair(self, pair: PairedExample) -> PairedExample:
        """Apply data augmentation to a pair."""
        # Augment text content
        if isinstance(pair.modality_a_content, str):
            pair.modality_a_content = self._augment_text(pair.modality_a_content)
        if isinstance(pair.modality_b_content, str):
            pair.modality_b_content = self._augment_text(pair.modality_b_content)

        return pair

    def _augment_text(self, text: str) -> str:
        """Apply text augmentation strategies."""
        if not text:
            return text

        augmentations = [
            self._random_word_dropout,
            self._random_word_swap,
            self._sentence_shuffle,
        ]

        # Apply one random augmentation with 50% probability
        if random.random() < 0.5:
            aug_fn = random.choice(augmentations)
            text = aug_fn(text)

        return text[: self.max_text_length]

    def _random_word_dropout(self, text: str, p: float = 0.1) -> str:
        """Randomly drop words."""
        words = text.split()
        if len(words) <= 3:
            return text
        words = [w for w in words if random.random() > p]
        return " ".join(words)

    def _random_word_swap(self, text: str, p: float = 0.1) -> str:
        """Randomly swap adjacent words."""
        words = text.split()
        if len(words) <= 2:
            return text

        for i in range(len(words) - 1):
            if random.random() < p:
                words[i], words[i + 1] = words[i + 1], words[i]

        return " ".join(words)

    def _sentence_shuffle(self, text: str) -> str:
        """Shuffle sentence order (for multi-sentence text)."""
        sentences = text.split(". ")
        if len(sentences) <= 1:
            return text
        random.shuffle(sentences)
        return ". ".join(sentences)

    def get_modality_distribution(self) -> dict[str, int]:
        """Get distribution of modality pairs."""
        dist = defaultdict(int)
        for pair in self.pairs:
            key = f"{pair.modality_a_type}_{pair.modality_b_type}"
            dist[key] += 1
        return dict(dist)


class HardNegativeDataset(CMMPairedDataset):
    """
    Dataset with hard negative mining for more effective contrastive learning.

    Hard negatives are items that are similar but not paired - they provide
    a stronger learning signal than random negatives.
    """

    def __init__(
        self,
        corpus_path: str,
        embedding_cache_path: str | None = None,
        num_hard_negatives: int = 5,
        **kwargs,
    ):
        super().__init__(corpus_path, **kwargs)

        self.num_hard_negatives = num_hard_negatives
        self.embedding_cache: dict[str, np.ndarray] = {}

        if embedding_cache_path and Path(embedding_cache_path).exists():
            self._load_embedding_cache(embedding_cache_path)

        # Build index for hard negative mining
        self._build_negative_index()

    def _load_embedding_cache(self, path: str):
        """Load pre-computed embeddings for hard negative mining."""
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.embedding_cache[data["id"]] = np.array(data["embedding"])
        logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")

    def _build_negative_index(self):
        """Build index for efficient hard negative sampling."""
        # Group pairs by modality type
        self.pairs_by_modality: dict[str, list[int]] = defaultdict(list)

        for i, pair in enumerate(self.pairs):
            self.pairs_by_modality[pair.modality_a_type].append(i)
            self.pairs_by_modality[pair.modality_b_type].append(i)

        # Build entity co-occurrence index for hard negatives
        self.pairs_by_entity: dict[str, list[int]] = defaultdict(list)
        for i, pair in enumerate(self.pairs):
            for entity in pair.bridging_entities:
                self.pairs_by_entity[entity.lower()].append(i)

    def get_hard_negatives(self, idx: int) -> list[PairedExample]:
        """
        Get hard negative examples for a given pair.

        Hard negatives share entities but are not the same pair.
        """
        pair = self.pairs[idx]
        candidates = set()

        # Find pairs sharing entities
        for entity in pair.bridging_entities:
            candidates.update(self.pairs_by_entity.get(entity.lower(), []))

        # Remove the pair itself
        candidates.discard(idx)

        # If not enough candidates from entities, add random from same modality
        if len(candidates) < self.num_hard_negatives:
            same_modality = set(self.pairs_by_modality.get(pair.modality_a_type, []))
            same_modality.discard(idx)
            candidates.update(same_modality)

        # Sample hard negatives
        candidates = list(candidates)
        if len(candidates) > self.num_hard_negatives:
            candidates = random.sample(candidates, self.num_hard_negatives)

        return [self.pairs[i] for i in candidates]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return pair with hard negatives."""
        pair = super().__getitem__(idx)
        hard_negatives = self.get_hard_negatives(idx)

        return {
            "positive": pair,
            "hard_negatives": hard_negatives,
        }


# =============================================================================
# Batch Samplers
# =============================================================================


class ModalityBalancedSampler(Sampler):
    """
    Sampler that ensures balanced representation of modality pairs in each batch.

    This prevents the model from overfitting to the most common modality pair.
    """

    def __init__(
        self,
        dataset: CMMPairedDataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by modality pair
        self.indices_by_modality: dict[str, list[int]] = defaultdict(list)
        for i, pair in enumerate(dataset.pairs):
            key = f"{pair.modality_a_type}_{pair.modality_b_type}"
            self.indices_by_modality[key].append(i)

        self.modality_keys = list(self.indices_by_modality.keys())

    def __iter__(self):
        # Shuffle within each modality group
        if self.shuffle:
            for key in self.modality_keys:
                random.shuffle(self.indices_by_modality[key])

        # Interleave modalities
        iterators = {k: iter(v) for k, v in self.indices_by_modality.items()}
        batch = []

        while iterators:
            # Round-robin through modalities
            keys_to_remove = []
            for key in list(iterators.keys()):
                try:
                    batch.append(next(iterators[key]))
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                except StopIteration:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del iterators[key]

        # Yield remaining
        if batch:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ConfidenceWeightedSampler(Sampler):
    """
    Sampler that weights examples by confidence score.

    Higher confidence pairs are sampled more frequently.
    """

    def __init__(
        self,
        dataset: CMMPairedDataset,
        num_samples: int,
        temperature: float = 1.0,
    ):
        self.dataset = dataset
        self.num_samples = num_samples
        self.temperature = temperature

        # Compute sampling weights from confidence scores
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute sampling weights from confidence scores."""
        confidences = np.array([p.confidence_score for p in self.dataset.pairs])

        # Apply temperature scaling
        weights = confidences ** (1.0 / self.temperature)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset), size=self.num_samples, replace=True, p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# =============================================================================
# Collation Functions
# =============================================================================


class ContrastiveCollator:
    """
    Collate function for contrastive learning batches.

    Handles different modalities appropriately:
    - Text: Tokenize and pad
    - Spectra: Convert to tensors and pad
    - Structures: Keep as list (processed by encoder)
    """

    def __init__(
        self,
        text_tokenizer=None,
        max_text_length: int = 512,
        spectrum_max_length: int = 8192,
    ):
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.spectrum_max_length = spectrum_max_length

    def __call__(self, batch: list[PairedExample]) -> ContrastiveBatch:
        """Collate a batch of paired examples."""

        modality_a_types = []
        modality_a_contents = []
        modality_b_types = []
        modality_b_contents = []
        pair_ids = []
        confidence_scores = []

        for pair in batch:
            modality_a_types.append(pair.modality_a_type)
            modality_a_contents.append(pair.modality_a_content)
            modality_b_types.append(pair.modality_b_type)
            modality_b_contents.append(pair.modality_b_content)
            pair_ids.append(pair.pair_id)
            confidence_scores.append(pair.confidence_score)

        # Create contrastive labels (diagonal is positive)
        batch_size = len(batch)
        labels = torch.arange(batch_size)

        # Convert confidence scores to tensor
        confidence_tensor = torch.tensor(confidence_scores, dtype=torch.float32)

        # Preprocess tensors if possible
        modality_a_tensors = self._preprocess_modality(modality_a_contents, modality_a_types)
        modality_b_tensors = self._preprocess_modality(modality_b_contents, modality_b_types)

        return ContrastiveBatch(
            modality_a_types=modality_a_types,
            modality_a_contents=modality_a_contents,
            modality_a_tensors=modality_a_tensors,
            modality_b_types=modality_b_types,
            modality_b_contents=modality_b_contents,
            modality_b_tensors=modality_b_tensors,
            labels=labels,
            pair_ids=pair_ids,
            confidence_scores=confidence_tensor,
        )

    def _preprocess_modality(self, contents: list[Any], types: list[str]) -> torch.Tensor | None:
        """
        Preprocess content based on modality type.

        Returns tensor if all items are same modality and can be batched,
        otherwise returns None.
        """
        # Check if all same type
        if len(set(types)) != 1:
            return None

        modality = types[0]

        # Text modalities
        if modality in ["text_scientific", "text_policy", "text_news"]:
            if self.text_tokenizer is None:
                return None

            # Tokenize text
            texts = [c if isinstance(c, str) else str(c) for c in contents]
            encoded = self.text_tokenizer(
                texts,
                max_length=self.max_text_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return encoded["input_ids"]

        # Spectrum modalities
        elif modality in ["spectrum_xrd", "spectrum_xrf", "spectrum_raman"]:
            try:
                tensors = []
                for content in contents:
                    if isinstance(content, dict) and "y" in content:
                        spectrum = np.array(content["y"], dtype=np.float32)
                    elif isinstance(content, (list, np.ndarray)):
                        spectrum = np.array(content, dtype=np.float32)
                    else:
                        return None

                    # Truncate/pad to max length
                    if len(spectrum) > self.spectrum_max_length:
                        spectrum = spectrum[: self.spectrum_max_length]
                    elif len(spectrum) < self.spectrum_max_length:
                        spectrum = np.pad(spectrum, (0, self.spectrum_max_length - len(spectrum)))

                    tensors.append(torch.from_numpy(spectrum))

                return torch.stack(tensors)
            except (RuntimeError, ValueError):
                return None

        return None


# =============================================================================
# DataLoader Factory
# =============================================================================


def create_contrastive_dataloader(
    dataset: CMMPairedDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    sampler_type: str = "random",
    text_tokenizer=None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for contrastive training.

    Args:
        dataset: CMMPairedDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (ignored if sampler is specified)
        sampler_type: One of "random", "balanced", "confidence"
        text_tokenizer: Optional tokenizer for text preprocessing
        **kwargs: Additional arguments for sampler

    Returns:
        DataLoader configured for contrastive learning
    """
    # Select sampler
    sampler = None
    if sampler_type == "balanced":
        sampler = ModalityBalancedSampler(dataset, batch_size, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling
    elif sampler_type == "confidence":
        num_samples = kwargs.get("num_samples", len(dataset))
        temperature = kwargs.get("temperature", 1.0)
        sampler = ConfidenceWeightedSampler(dataset, num_samples, temperature)
        shuffle = False

    # Create collator
    collator = ContrastiveCollator(
        text_tokenizer=text_tokenizer,
        max_text_length=kwargs.get("max_text_length", 512),
        spectrum_max_length=kwargs.get("spectrum_max_length", 8192),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size if sampler_type == "random" else 1,  # Sampler handles batching
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator
        if sampler_type == "random"
        else lambda x: collator(x[0]) if isinstance(x[0], list) else collator(x),
        pin_memory=True,
        drop_last=True,  # Important for contrastive learning
    )


def create_evaluation_dataloader(
    corpus_path: str,
    batch_size: int = 32,
    modality_filter: tuple[str, str] | None = None,
) -> DataLoader:
    """Create a DataLoader for evaluation (no augmentation, no shuffling)."""
    dataset = CMMPairedDataset(
        corpus_path,
        min_confidence=0.0,
        modality_filter=modality_filter,
        augment=False,
    )

    collator = ContrastiveCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=True,
    )


# =============================================================================
# Training Utilities
# =============================================================================


class InBatchNegativeSampler:
    """
    Utility for in-batch negative sampling during contrastive training.

    In contrastive learning, other items in the same batch serve as negatives.
    This class provides utilities for computing the similarity matrix and loss.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def compute_similarity_matrix(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between two sets of embeddings.

        Args:
            embeddings_a: Shape (batch_size, embedding_dim)
            embeddings_b: Shape (batch_size, embedding_dim)

        Returns:
            Similarity matrix of shape (batch_size, batch_size)
        """
        # Normalize embeddings
        embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        return similarity

    def compute_contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        confidence_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute symmetric contrastive loss (InfoNCE).

        Positive pairs are on the diagonal of the similarity matrix.
        """
        batch_size = embeddings_a.size(0)
        labels = torch.arange(batch_size, device=embeddings_a.device)

        similarity = self.compute_similarity_matrix(embeddings_a, embeddings_b)

        # Symmetric loss
        loss_a_to_b = torch.nn.functional.cross_entropy(similarity, labels, reduction="none")
        loss_b_to_a = torch.nn.functional.cross_entropy(similarity.T, labels, reduction="none")

        loss = (loss_a_to_b + loss_b_to_a) / 2

        # Apply confidence weighting if provided
        if confidence_weights is not None:
            loss = loss * confidence_weights

        return loss.mean()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("corpus_path", help="Path to corpus JSONL file")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Create dataset
    dataset = CMMPairedDataset(
        args.corpus_path,
        min_confidence=0.3,
        augment=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Modality distribution: {dataset.get_modality_distribution()}")

    # Create dataloader
    dataloader = create_contrastive_dataloader(
        dataset,
        batch_size=args.batch_size,
        sampler_type="balanced",
    )

    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Modality A types: {batch.modality_a_types}")
        print(f"  Modality B types: {batch.modality_b_types}")
        print(f"  Confidence scores: {batch.confidence_scores}")

        if i >= 2:
            break
