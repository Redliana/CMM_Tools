from __future__ import annotations

# embedding/training/corpus_builder.py
"""
CMM Training Corpus Builder for Cross-Modal Alignment

This module provides tools for constructing paired training data across modalities
for contrastive learning. It addresses the key challenge that no existing corpus
pairs atomistic simulations with policy documents.

Strategies implemented:
1. Direct pairing from structured databases (Materials Project, ICSD)
2. LLM-based synthetic pairing for bridging descriptions
3. Weak supervision using entity co-occurrence
4. Human-in-the-loop curation workflow

Usage:
    builder = CMMCorpusBuilder(config)
    corpus = builder.build_corpus()
    corpus.save("cmm_training_corpus.jsonl")
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================


class Modality(Enum):
    """Supported modalities for cross-modal alignment."""

    TEXT_SCIENTIFIC = "text_scientific"
    TEXT_POLICY = "text_policy"
    TEXT_NEWS = "text_news"
    SPECTRUM_XRD = "spectrum_xrd"
    SPECTRUM_XRF = "spectrum_xrf"
    SPECTRUM_RAMAN = "spectrum_raman"
    MOLECULAR_STRUCTURE = "molecular_structure"
    CRYSTAL_STRUCTURE = "crystal_structure"
    TABULAR_TRADE = "tabular_trade"
    TABULAR_PRODUCTION = "tabular_production"
    GEOSPATIAL = "geospatial"


class PairingMethod(Enum):
    """Method used to create the pair."""

    DIRECT_DATABASE = "direct_database"  # From structured DB relationships
    LLM_SYNTHETIC = "llm_synthetic"  # LLM-generated bridging text
    ENTITY_COOCCURRENCE = "entity_cooccurrence"  # Weak supervision
    CITATION_LINK = "citation_link"  # Academic citation relationships
    HUMAN_CURATED = "human_curated"  # Expert annotation
    METADATA_MATCH = "metadata_match"  # Matching metadata fields


@dataclass
class ModalityData:
    """Container for data from a single modality."""

    modality: Modality
    content: Any  # Raw content (text, spectrum array, structure dict, etc.)
    content_hash: str  # For deduplication
    source: str  # Data source identifier
    source_id: str  # ID within the source
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        if isinstance(self.content, str):
            content_bytes = self.content.encode("utf-8")
        elif isinstance(self.content, (list, dict)):
            content_bytes = json.dumps(self.content, sort_keys=True).encode("utf-8")
        elif isinstance(self.content, np.ndarray):
            content_bytes = self.content.tobytes()
        else:
            content_bytes = str(self.content).encode("utf-8")
        return hashlib.sha256(content_bytes).hexdigest()[:16]


@dataclass
class CrossModalPair:
    """A paired example for contrastive learning."""

    pair_id: str
    modality_a: ModalityData
    modality_b: ModalityData
    pairing_method: PairingMethod
    confidence_score: float  # 0.0 to 1.0
    bridging_entities: list[str] = field(default_factory=list)  # CMM entities linking the pair
    bridging_text: str | None = None  # LLM-generated explanation of relationship
    human_validated: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "modality_a": {
                "modality": self.modality_a.modality.value,
                "content": self.modality_a.content
                if isinstance(self.modality_a.content, (str, dict, list))
                else None,
                "content_hash": self.modality_a.content_hash,
                "source": self.modality_a.source,
                "source_id": self.modality_a.source_id,
                "metadata": self.modality_a.metadata,
            },
            "modality_b": {
                "modality": self.modality_b.modality.value,
                "content": self.modality_b.content
                if isinstance(self.modality_b.content, (str, dict, list))
                else None,
                "content_hash": self.modality_b.content_hash,
                "source": self.modality_b.source,
                "source_id": self.modality_b.source_id,
                "metadata": self.modality_b.metadata,
            },
            "pairing_method": self.pairing_method.value,
            "confidence_score": self.confidence_score,
            "bridging_entities": self.bridging_entities,
            "bridging_text": self.bridging_text,
            "human_validated": self.human_validated,
            "created_at": self.created_at,
        }


@dataclass
class TrainingCorpus:
    """Collection of cross-modal pairs for training."""

    pairs: list[CrossModalPair] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_pair(self, pair: CrossModalPair):
        self.pairs.append(pair)

    def filter_by_confidence(self, min_confidence: float) -> TrainingCorpus:
        """Return corpus filtered by minimum confidence."""
        filtered = TrainingCorpus(
            pairs=[p for p in self.pairs if p.confidence_score >= min_confidence],
            metadata={**self.metadata, "filtered_min_confidence": min_confidence},
        )
        return filtered

    def filter_by_modalities(self, modality_a: Modality, modality_b: Modality) -> TrainingCorpus:
        """Return corpus filtered to specific modality pair."""
        filtered_pairs = []
        for p in self.pairs:
            if (p.modality_a.modality == modality_a and p.modality_b.modality == modality_b) or (
                p.modality_a.modality == modality_b and p.modality_b.modality == modality_a
            ):
                filtered_pairs.append(p)
        return TrainingCorpus(
            pairs=filtered_pairs,
            metadata={**self.metadata, "filtered_modalities": [modality_a.value, modality_b.value]},
        )

    def get_statistics(self) -> dict[str, Any]:
        """Compute corpus statistics."""
        modality_counts = {}
        method_counts = {}
        confidence_scores = []

        for pair in self.pairs:
            # Count modalities
            for mod in [pair.modality_a.modality, pair.modality_b.modality]:
                modality_counts[mod.value] = modality_counts.get(mod.value, 0) + 1

            # Count methods
            method_counts[pair.pairing_method.value] = (
                method_counts.get(pair.pairing_method.value, 0) + 1
            )

            # Collect confidence scores
            confidence_scores.append(pair.confidence_score)

        return {
            "total_pairs": len(self.pairs),
            "modality_distribution": modality_counts,
            "pairing_method_distribution": method_counts,
            "confidence_mean": np.mean(confidence_scores) if confidence_scores else 0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0,
            "human_validated_count": sum(1 for p in self.pairs if p.human_validated),
        }

    def save(self, path: str):
        """Save corpus to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Write metadata as first line
            f.write(
                json.dumps(
                    {
                        "_metadata": {
                            **self.metadata,
                            "statistics": self.get_statistics(),
                            "saved_at": datetime.utcnow().isoformat(),
                        }
                    }
                )
                + "\n"
            )

            # Write pairs
            for pair in self.pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        logger.info(f"Saved {len(self.pairs)} pairs to {path}")

    @classmethod
    def load(cls, path: str) -> TrainingCorpus:
        """Load corpus from JSONL file."""
        corpus = cls()

        with open(path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if i == 0 and "_metadata" in data:
                    corpus.metadata = data["_metadata"]
                else:
                    # Reconstruct pair (simplified - content may need special handling)
                    pair = CrossModalPair(
                        pair_id=data["pair_id"],
                        modality_a=ModalityData(
                            modality=Modality(data["modality_a"]["modality"]),
                            content=data["modality_a"]["content"],
                            content_hash=data["modality_a"]["content_hash"],
                            source=data["modality_a"]["source"],
                            source_id=data["modality_a"]["source_id"],
                            metadata=data["modality_a"]["metadata"],
                        ),
                        modality_b=ModalityData(
                            modality=Modality(data["modality_b"]["modality"]),
                            content=data["modality_b"]["content"],
                            content_hash=data["modality_b"]["content_hash"],
                            source=data["modality_b"]["source"],
                            source_id=data["modality_b"]["source_id"],
                            metadata=data["modality_b"]["metadata"],
                        ),
                        pairing_method=PairingMethod(data["pairing_method"]),
                        confidence_score=data["confidence_score"],
                        bridging_entities=data.get("bridging_entities", []),
                        bridging_text=data.get("bridging_text"),
                        human_validated=data.get("human_validated", False),
                        created_at=data.get("created_at", ""),
                    )
                    corpus.add_pair(pair)

        logger.info(f"Loaded {len(corpus.pairs)} pairs from {path}")
        return corpus


# =============================================================================
# Data Source Connectors
# =============================================================================


class DataSourceConnector(ABC):
    """Abstract base class for data source connectors."""

    @abstractmethod
    async def fetch_items(self, limit: int | None = None) -> Iterator[ModalityData]:
        """Fetch items from the data source."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the source name."""
        pass


class MaterialsProjectConnector(DataSourceConnector):
    """Connector for Materials Project API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org/v2"

    def get_source_name(self) -> str:
        return "materials_project"

    async def fetch_items(self, limit: int = 1000) -> Iterator[ModalityData]:
        """
        Fetch crystal structures and computed properties from Materials Project.

        Returns ModalityData for both crystal structures and associated text descriptions.
        """
        import httpx

        # CMM-relevant elements
        cmm_elements = [
            "Li",
            "Co",
            "Ni",
            "Mn",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Nb",
            "Mo",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "In",
            "Sn",
            "Sb",
            "Te",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Bi",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
        ]

        async with httpx.AsyncClient() as client:
            headers = {"X-API-KEY": self.api_key}

            for element in cmm_elements[:5]:  # Limit for demo
                params = {
                    "elements": element,
                    "fields": "material_id,formula_pretty,structure,band_gap,formation_energy_per_atom",
                    "limit": limit // len(cmm_elements),
                }

                try:
                    response = await client.get(
                        f"{self.base_url}/materials/summary",
                        headers=headers,
                        params=params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    for material in data.get("data", []):
                        # Yield crystal structure
                        if material.get("structure"):
                            yield ModalityData(
                                modality=Modality.CRYSTAL_STRUCTURE,
                                content=material["structure"],
                                content_hash="",
                                source=self.get_source_name(),
                                source_id=material["material_id"],
                                metadata={
                                    "formula": material.get("formula_pretty"),
                                    "band_gap": material.get("band_gap"),
                                    "formation_energy": material.get("formation_energy_per_atom"),
                                },
                            )

                        # Generate text description
                        text_desc = self._generate_text_description(material)
                        if text_desc:
                            yield ModalityData(
                                modality=Modality.TEXT_SCIENTIFIC,
                                content=text_desc,
                                content_hash="",
                                source=self.get_source_name(),
                                source_id=f"{material['material_id']}_text",
                                metadata={
                                    "formula": material.get("formula_pretty"),
                                    "related_structure_id": material["material_id"],
                                },
                            )

                except httpx.HTTPError as e:
                    logger.error(f"Error fetching from Materials Project: {e}")
                    continue

    def _generate_text_description(self, material: dict) -> str | None:
        """Generate text description from material properties."""
        formula = material.get("formula_pretty", "Unknown")
        band_gap = material.get("band_gap")
        formation_energy = material.get("formation_energy_per_atom")

        parts = [f"{formula} is a crystalline material"]

        if band_gap is not None:
            if band_gap == 0:
                parts.append("with metallic character (zero band gap)")
            elif band_gap < 1.0:
                parts.append(
                    f"with a narrow band gap of {band_gap:.2f} eV, suitable for infrared applications"
                )
            elif band_gap < 3.0:
                parts.append(
                    f"with a band gap of {band_gap:.2f} eV, potentially useful for visible-light applications"
                )
            else:
                parts.append(
                    f"with a wide band gap of {band_gap:.2f} eV, indicating insulating behavior"
                )

        if formation_energy is not None:
            if formation_energy < -1.0:
                parts.append(
                    f"The formation energy of {formation_energy:.3f} eV/atom indicates high thermodynamic stability."
                )
            elif formation_energy < 0:
                parts.append(
                    f"The formation energy of {formation_energy:.3f} eV/atom suggests moderate stability."
                )
            else:
                parts.append(
                    f"The positive formation energy of {formation_energy:.3f} eV/atom indicates metastability."
                )

        return " ".join(parts) if len(parts) > 1 else None


class USGSConnector(DataSourceConnector):
    """Connector for USGS Mineral Resources data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_source_name(self) -> str:
        return "usgs"

    async def fetch_items(self, limit: int | None = None) -> Iterator[ModalityData]:
        """
        Fetch USGS mineral commodity summaries and reports.

        Expects pre-downloaded PDFs converted to text in data_dir.
        """
        text_files = list(self.data_dir.glob("**/*.txt"))

        for i, text_file in enumerate(text_files):
            if limit and i >= limit:
                break

            try:
                content = text_file.read_text(encoding="utf-8")

                # Determine if scientific or policy based on content
                modality = (
                    Modality.TEXT_POLICY
                    if any(
                        term in content.lower()
                        for term in [
                            "policy",
                            "regulation",
                            "import",
                            "export",
                            "tariff",
                            "stockpile",
                        ]
                    )
                    else Modality.TEXT_SCIENTIFIC
                )

                yield ModalityData(
                    modality=modality,
                    content=content,
                    content_hash="",
                    source=self.get_source_name(),
                    source_id=text_file.stem,
                    metadata={
                        "filename": text_file.name,
                        "year": self._extract_year(text_file.name),
                    },
                )
            except OSError as e:
                logger.error(f"Error reading {text_file}: {e}")
                continue

    def _extract_year(self, filename: str) -> int | None:
        """Extract year from filename if present."""
        import re

        match = re.search(r"20\d{2}", filename)
        return int(match.group()) if match else None


class FederalRegisterConnector(DataSourceConnector):
    """Connector for Federal Register API (policy documents)."""

    def __init__(self):
        self.base_url = "https://www.federalregister.gov/api/v1"

    def get_source_name(self) -> str:
        return "federal_register"

    async def fetch_items(self, limit: int = 100) -> Iterator[ModalityData]:
        """Fetch CMM-related policy documents from Federal Register."""
        import httpx

        # CMM-related search terms
        search_terms = [
            "critical minerals",
            "rare earth elements",
            "cobalt import",
            "lithium battery",
            "semiconductor materials",
            "defense production act minerals",
            "section 232 minerals",
        ]

        async with httpx.AsyncClient() as client:
            for term in search_terms:
                try:
                    params = {
                        "conditions[term]": term,
                        "per_page": min(limit // len(search_terms), 100),
                        "order": "relevance",
                    }

                    response = await client.get(
                        f"{self.base_url}/documents.json", params=params, timeout=30.0
                    )
                    response.raise_for_status()
                    data = response.json()

                    for doc in data.get("results", []):
                        yield ModalityData(
                            modality=Modality.TEXT_POLICY,
                            content=doc.get("abstract", "") or doc.get("title", ""),
                            content_hash="",
                            source=self.get_source_name(),
                            source_id=doc.get("document_number", ""),
                            metadata={
                                "title": doc.get("title"),
                                "publication_date": doc.get("publication_date"),
                                "document_type": doc.get("type"),
                                "agencies": [a.get("name") for a in doc.get("agencies", [])],
                                "full_text_url": doc.get("full_text_xml_url"),
                            },
                        )

                except httpx.HTTPError as e:
                    logger.error(f"Error fetching from Federal Register for '{term}': {e}")
                    continue


class SpectrumDatabaseConnector(DataSourceConnector):
    """Connector for spectral databases (RRUFF, etc.)."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_source_name(self) -> str:
        return "spectrum_database"

    async def fetch_items(self, limit: int | None = None) -> Iterator[ModalityData]:
        """
        Fetch spectral data from local database.

        Expects spectrum files in formats: .xy (XRD), .txt (Raman), etc.
        """
        spectrum_files = list(self.data_dir.glob("**/*"))

        for i, spec_file in enumerate(spectrum_files):
            if limit and i >= limit:
                break

            try:
                # Determine modality from file extension/content
                modality = self._determine_spectrum_modality(spec_file)
                if modality is None:
                    continue

                # Load spectrum data
                spectrum_data = self._load_spectrum(spec_file)
                if spectrum_data is None:
                    continue

                yield ModalityData(
                    modality=modality,
                    content=spectrum_data,
                    content_hash="",
                    source=self.get_source_name(),
                    source_id=spec_file.stem,
                    metadata={
                        "filename": spec_file.name,
                        "mineral_name": self._extract_mineral_name(spec_file.stem),
                    },
                )
            except (OSError, ValueError) as e:
                logger.error(f"Error loading spectrum {spec_file}: {e}")
                continue

    def _determine_spectrum_modality(self, path: Path) -> Modality | None:
        """Determine spectrum modality from file."""
        suffix = path.suffix.lower()
        name = path.name.lower()

        if "xrd" in name or suffix in [".xy", ".raw"]:
            return Modality.SPECTRUM_XRD
        elif "raman" in name:
            return Modality.SPECTRUM_RAMAN
        elif "xrf" in name:
            return Modality.SPECTRUM_XRF
        return None

    def _load_spectrum(self, path: Path) -> dict | None:
        """Load spectrum data from file."""
        try:
            # Simple two-column format (x, y)
            data = np.loadtxt(path, comments=["#", ";"])
            if data.ndim == 2 and data.shape[1] >= 2:
                return {
                    "x": data[:, 0].tolist(),
                    "y": data[:, 1].tolist(),
                }
        except (OSError, ValueError):
            pass
        return None

    def _extract_mineral_name(self, stem: str) -> str | None:
        """Extract mineral name from filename."""
        # Remove common suffixes like _XRD, _001, etc.
        import re

        name = re.sub(r"[_-]?(xrd|raman|xrf|[0-9]+)$", "", stem, flags=re.IGNORECASE)
        return name if name else None


# =============================================================================
# Pairing Strategies
# =============================================================================


class PairingStrategy(ABC):
    """Abstract base class for pairing strategies."""

    @abstractmethod
    async def generate_pairs(
        self,
        items_a: list[ModalityData],
        items_b: list[ModalityData],
    ) -> Iterator[CrossModalPair]:
        """Generate cross-modal pairs from two lists of items."""
        pass

    @abstractmethod
    def get_method(self) -> PairingMethod:
        """Return the pairing method."""
        pass


class EntityCooccurrenceStrategy(PairingStrategy):
    """
    Weak supervision strategy: pair items that mention the same CMM entities.

    This is a high-recall, lower-precision strategy suitable for initial corpus building.
    """

    def __init__(self, entity_list: list[str], min_overlap: int = 1):
        """
        Args:
            entity_list: list of CMM entities to look for
            min_overlap: Minimum number of shared entities to create a pair
        """
        self.entity_list = [e.lower() for e in entity_list]
        self.min_overlap = min_overlap

    def get_method(self) -> PairingMethod:
        return PairingMethod.ENTITY_COOCCURRENCE

    def _extract_entities(self, item: ModalityData) -> set:
        """Extract CMM entities mentioned in the item."""
        if isinstance(item.content, str):
            text = item.content.lower()
        elif isinstance(item.content, dict):
            text = json.dumps(item.content).lower()
        else:
            text = str(item.content).lower()

        return {e for e in self.entity_list if e in text}

    async def generate_pairs(
        self,
        items_a: list[ModalityData],
        items_b: list[ModalityData],
    ) -> Iterator[CrossModalPair]:
        """Generate pairs based on entity co-occurrence."""

        # Pre-compute entities for all items
        entities_a = [(item, self._extract_entities(item)) for item in items_a]
        entities_b = [(item, self._extract_entities(item)) for item in items_b]

        pair_count = 0
        for item_a, ents_a in entities_a:
            if not ents_a:
                continue

            for item_b, ents_b in entities_b:
                if not ents_b:
                    continue

                overlap = ents_a & ents_b
                if len(overlap) >= self.min_overlap:
                    # Confidence based on overlap ratio
                    confidence = len(overlap) / max(len(ents_a), len(ents_b))

                    yield CrossModalPair(
                        pair_id=f"eco_{pair_count:06d}",
                        modality_a=item_a,
                        modality_b=item_b,
                        pairing_method=self.get_method(),
                        confidence_score=min(
                            confidence * 0.8, 0.8
                        ),  # Cap at 0.8 for weak supervision
                        bridging_entities=list(overlap),
                    )
                    pair_count += 1


class LLMSyntheticPairingStrategy(PairingStrategy):
    """
    Use LLM to generate bridging descriptions and validate pairs.

    This strategy:
    1. Takes a pair of items from different modalities
    2. Asks LLM to generate a bridging description explaining their relationship
    3. Uses LLM confidence to score the pair
    """

    def __init__(
        self,
        llm_client,  # Anthropic or OpenAI client
        model: str = "claude-sonnet-4-20250514",
        batch_size: int = 10,
    ):
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size

    def get_method(self) -> PairingMethod:
        return PairingMethod.LLM_SYNTHETIC

    def _format_item_for_prompt(self, item: ModalityData) -> str:
        """Format an item for inclusion in LLM prompt."""
        if item.modality in [Modality.TEXT_SCIENTIFIC, Modality.TEXT_POLICY, Modality.TEXT_NEWS]:
            # Truncate long text
            content = (
                item.content[:2000] if isinstance(item.content, str) else str(item.content)[:2000]
            )
            return f"[{item.modality.value}]\n{content}"

        elif item.modality == Modality.CRYSTAL_STRUCTURE:
            # Summarize structure
            if isinstance(item.content, dict):
                formula = item.metadata.get("formula", "Unknown")
                return f"[Crystal Structure]\nFormula: {formula}\nProperties: {json.dumps(item.metadata, indent=2)}"
            return f"[Crystal Structure]\n{str(item.content)[:500]}"

        elif item.modality in [
            Modality.SPECTRUM_XRD,
            Modality.SPECTRUM_XRF,
            Modality.SPECTRUM_RAMAN,
        ]:
            mineral = item.metadata.get("mineral_name", "Unknown")
            return f"[{item.modality.value}]\nMineral: {mineral}\nSource: {item.source}"

        else:
            return f"[{item.modality.value}]\n{str(item.content)[:500]}"

    async def _generate_bridging_text(
        self,
        item_a: ModalityData,
        item_b: ModalityData,
    ) -> tuple[str | None, float]:
        """
        Use LLM to generate bridging text and assess relationship strength.

        Returns:
            tuple of (bridging_text, confidence_score)
        """
        prompt = f"""You are an expert in critical minerals and materials (CMM) supply chains.

Given two pieces of information from different modalities, your task is to:
1. Determine if there is a meaningful relationship between them in the context of CMM
2. If yes, write a brief bridging description (1-2 sentences) explaining the relationship
3. Rate your confidence in this relationship from 0.0 to 1.0

Item A:
{self._format_item_for_prompt(item_a)}

Item B:
{self._format_item_for_prompt(item_b)}

Respond in JSON format:
{{
    "has_relationship": true/false,
    "bridging_description": "Description of how these items relate in the CMM context...",
    "confidence": 0.0-1.0,
    "key_entities": ["entity1", "entity2", ...]
}}

If there is no meaningful CMM-related relationship, set has_relationship to false and confidence to 0."""

        try:
            # Using Anthropic client
            response = await self.llm_client.messages.create(
                model=self.model, max_tokens=500, messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = response.content[0].text
            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("has_relationship"):
                    return (result.get("bridging_description"), result.get("confidence", 0.5))
            return None, 0.0

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"LLM bridging generation failed: {e}")
            return None, 0.0

    async def generate_pairs(
        self,
        items_a: list[ModalityData],
        items_b: list[ModalityData],
    ) -> Iterator[CrossModalPair]:
        """Generate pairs using LLM to validate relationships."""

        pair_count = 0

        # Process in batches
        for i in range(0, len(items_a), self.batch_size):
            batch_a = items_a[i : i + self.batch_size]

            for item_a in batch_a:
                for item_b in items_b[: self.batch_size]:  # Limit comparisons
                    bridging_text, confidence = await self._generate_bridging_text(item_a, item_b)

                    if bridging_text and confidence > 0.3:
                        yield CrossModalPair(
                            pair_id=f"llm_{pair_count:06d}",
                            modality_a=item_a,
                            modality_b=item_b,
                            pairing_method=self.get_method(),
                            confidence_score=confidence,
                            bridging_text=bridging_text,
                        )
                        pair_count += 1


class MetadataMatchStrategy(PairingStrategy):
    """
    Pair items based on matching metadata fields (e.g., same mineral, same material ID).

    High precision for items from structured databases.
    """

    def __init__(self, match_fields: list[str]):
        """
        Args:
            match_fields: list of metadata field names to match on
        """
        self.match_fields = match_fields

    def get_method(self) -> PairingMethod:
        return PairingMethod.METADATA_MATCH

    async def generate_pairs(
        self,
        items_a: list[ModalityData],
        items_b: list[ModalityData],
    ) -> Iterator[CrossModalPair]:
        """Generate pairs based on metadata field matches."""

        pair_count = 0

        for item_a in items_a:
            for item_b in items_b:
                # Check for matching fields
                matched_fields = []
                for field in self.match_fields:
                    val_a = item_a.metadata.get(field)
                    val_b = item_b.metadata.get(field)
                    if val_a and val_b and str(val_a).lower() == str(val_b).lower():
                        matched_fields.append(field)

                if matched_fields:
                    confidence = len(matched_fields) / len(self.match_fields)

                    yield CrossModalPair(
                        pair_id=f"meta_{pair_count:06d}",
                        modality_a=item_a,
                        modality_b=item_b,
                        pairing_method=self.get_method(),
                        confidence_score=min(confidence, 0.95),
                        bridging_entities=matched_fields,
                    )
                    pair_count += 1


# =============================================================================
# Main Corpus Builder
# =============================================================================


@dataclass
class CorpusBuilderConfig:
    """Configuration for corpus building."""

    output_dir: str = "./cmm_corpus"
    materials_project_api_key: str | None = None
    usgs_data_dir: str | None = None
    spectrum_data_dir: str | None = None
    llm_client: Any | None = None

    # CMM entity list for weak supervision
    cmm_entities: list[str] = field(
        default_factory=lambda: [
            "lithium",
            "cobalt",
            "nickel",
            "manganese",
            "graphite",
            "copper",
            "rare earth",
            "neodymium",
            "dysprosium",
            "terbium",
            "praseodymium",
            "gallium",
            "germanium",
            "indium",
            "tungsten",
            "tantalum",
            "niobium",
            "antimony",
            "bismuth",
            "vanadium",
            "titanium",
            "platinum",
            "palladium",
            "DRC",
            "Congo",
            "Australia",
            "Chile",
            "China",
            "Indonesia",
            "battery",
            "semiconductor",
            "magnet",
            "catalyst",
            "alloy",
            "mining",
            "processing",
            "refining",
            "recycling",
            "supply chain",
        ]
    )


class CMMCorpusBuilder:
    """
    Main class for building the CMM cross-modal training corpus.

    Usage:
        config = CorpusBuilderConfig(
            materials_project_api_key="your_key",
            usgs_data_dir="/path/to/usgs",
        )
        builder = CMMCorpusBuilder(config)
        corpus = await builder.build_corpus()
        corpus.save("cmm_training_corpus.jsonl")
    """

    def __init__(self, config: CorpusBuilderConfig):
        self.config = config
        self.connectors: list[DataSourceConnector] = []
        self.strategies: list[PairingStrategy] = []

        self._setup_connectors()
        self._setup_strategies()

    def _setup_connectors(self):
        """Initialize data source connectors based on config."""
        if self.config.materials_project_api_key:
            self.connectors.append(MaterialsProjectConnector(self.config.materials_project_api_key))

        if self.config.usgs_data_dir:
            self.connectors.append(USGSConnector(self.config.usgs_data_dir))

        if self.config.spectrum_data_dir:
            self.connectors.append(SpectrumDatabaseConnector(self.config.spectrum_data_dir))

        # Always add Federal Register (public API)
        self.connectors.append(FederalRegisterConnector())

    def _setup_strategies(self):
        """Initialize pairing strategies."""
        # Entity co-occurrence (weak supervision)
        self.strategies.append(EntityCooccurrenceStrategy(self.config.cmm_entities, min_overlap=2))

        # Metadata matching
        self.strategies.append(MetadataMatchStrategy(["formula", "mineral_name", "material_id"]))

        # LLM synthetic pairing (if client provided)
        if self.config.llm_client:
            self.strategies.append(LLMSyntheticPairingStrategy(self.config.llm_client))

    async def _fetch_all_items(self) -> dict[Modality, list[ModalityData]]:
        """Fetch items from all connectors, grouped by modality."""
        items_by_modality: dict[Modality, list[ModalityData]] = {m: [] for m in Modality}

        for connector in self.connectors:
            logger.info(f"Fetching from {connector.get_source_name()}...")
            async for item in connector.fetch_items():
                items_by_modality[item.modality].append(item)

        # Log statistics
        for modality, items in items_by_modality.items():
            if items:
                logger.info(f"  {modality.value}: {len(items)} items")

        return items_by_modality

    async def build_corpus(self) -> TrainingCorpus:
        """
        Build the complete cross-modal training corpus.

        Returns:
            TrainingCorpus with all generated pairs
        """
        logger.info("Starting corpus build...")

        # Step 1: Fetch all items
        items_by_modality = await self._fetch_all_items()

        # Step 2: Generate pairs across modalities
        corpus = TrainingCorpus(
            metadata={
                "build_started": datetime.utcnow().isoformat(),
                "config": asdict(self.config)
                if hasattr(self.config, "__dataclass_fields__")
                else {},
            }
        )

        # Define modality pairs to generate
        modality_pairs = [
            # Scientific text <-> Crystal structure
            (Modality.TEXT_SCIENTIFIC, Modality.CRYSTAL_STRUCTURE),
            # Scientific text <-> Spectra
            (Modality.TEXT_SCIENTIFIC, Modality.SPECTRUM_XRD),
            (Modality.TEXT_SCIENTIFIC, Modality.SPECTRUM_XRF),
            # Policy text <-> Scientific text (scale bridging)
            (Modality.TEXT_POLICY, Modality.TEXT_SCIENTIFIC),
            # Crystal structure <-> Spectra
            (Modality.CRYSTAL_STRUCTURE, Modality.SPECTRUM_XRD),
        ]

        for mod_a, mod_b in modality_pairs:
            items_a = items_by_modality.get(mod_a, [])
            items_b = items_by_modality.get(mod_b, [])

            if not items_a or not items_b:
                logger.info(f"Skipping {mod_a.value} <-> {mod_b.value}: insufficient data")
                continue

            logger.info(f"Generating pairs: {mod_a.value} <-> {mod_b.value}")

            for strategy in self.strategies:
                async for pair in strategy.generate_pairs(items_a, items_b):
                    corpus.add_pair(pair)

        # Step 3: Log statistics
        stats = corpus.get_statistics()
        logger.info(f"Corpus built: {stats['total_pairs']} pairs")
        logger.info(f"  Method distribution: {stats['pairing_method_distribution']}")
        logger.info(f"  Mean confidence: {stats['confidence_mean']:.3f}")

        corpus.metadata["build_completed"] = datetime.utcnow().isoformat()

        return corpus


# =============================================================================
# Utility Functions
# =============================================================================


def create_default_corpus_builder() -> CMMCorpusBuilder:
    """Create a corpus builder with default configuration."""
    import os

    config = CorpusBuilderConfig(
        output_dir="./cmm_corpus",
        materials_project_api_key=os.environ.get("MP_API_KEY"),
        usgs_data_dir=os.environ.get("USGS_DATA_DIR"),
        spectrum_data_dir=os.environ.get("SPECTRUM_DATA_DIR"),
    )

    return CMMCorpusBuilder(config)


async def build_sample_corpus(output_path: str = "sample_corpus.jsonl"):
    """Build a small sample corpus for testing."""
    config = CorpusBuilderConfig()
    builder = CMMCorpusBuilder(config)
    corpus = await builder.build_corpus()
    corpus.save(output_path)
    return corpus


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CMM cross-modal training corpus")
    parser.add_argument("--output", "-o", default="cmm_corpus.jsonl", help="Output file path")
    parser.add_argument("--mp-key", help="Materials Project API key")
    parser.add_argument("--usgs-dir", help="USGS data directory")
    parser.add_argument("--spectrum-dir", help="Spectrum data directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = CorpusBuilderConfig(
        materials_project_api_key=args.mp_key,
        usgs_data_dir=args.usgs_dir,
        spectrum_data_dir=args.spectrum_dir,
    )

    builder = CMMCorpusBuilder(config)
    corpus = asyncio.run(builder.build_corpus())
    corpus.save(args.output)

    print(f"\nCorpus saved to {args.output}")
    print(f"Statistics: {json.dumps(corpus.get_statistics(), indent=2)}")
