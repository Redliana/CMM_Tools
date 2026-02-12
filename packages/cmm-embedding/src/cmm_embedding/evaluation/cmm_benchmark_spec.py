from __future__ import annotations

# embedding/evaluation/cmm_benchmark_spec.py
"""
CMM Embedding Evaluation Benchmark Specification

This module defines the comprehensive evaluation framework for CMM embeddings,
addressing the key gap that no CMM-specific embedding benchmark exists.

The benchmark evaluates five core capabilities:
1. Cross-Scale Retrieval - Link atomistic data to policy documents
2. Cross-Modal Alignment - Retrieve across modalities (spectra ↔ text)
3. Entity Resolution - Match entity mentions across naming variations
4. Supply Chain Traversal - Multi-hop queries through supply chain graph
5. Temporal Consistency - Handle time-sensitive information correctly

Reference: CMM_RAG_Implementation_Guide.md Section 12.5
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Benchmark Categories
# =============================================================================


class BenchmarkCategory(Enum):
    """Categories of CMM benchmark tasks."""

    CROSS_SCALE_RETRIEVAL = "cross_scale_retrieval"
    CROSS_MODAL_ALIGNMENT = "cross_modal_alignment"
    ENTITY_RESOLUTION = "entity_resolution"
    SUPPLY_CHAIN_TRAVERSAL = "supply_chain_traversal"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class Difficulty(Enum):
    """Difficulty levels for benchmark tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ScaleLevel(Enum):
    """Scale levels in the CMM hierarchy."""

    ATOMISTIC = "atomistic"  # DFT calculations, molecular properties
    MATERIAL = "material"  # Bulk material properties, synthesis
    FACILITY = "facility"  # Mine/processor/plant level
    REGIONAL = "regional"  # District/province level
    NATIONAL = "national"  # Country-level policies, production
    GLOBAL = "global"  # International trade, supply chains


class ModalityType(Enum):
    """Modality types for cross-modal evaluation."""

    TEXT_SCIENTIFIC = "text_scientific"
    TEXT_POLICY = "text_policy"
    TEXT_NEWS = "text_news"
    SPECTRUM_XRD = "spectrum_xrd"
    SPECTRUM_XRF = "spectrum_xrf"
    CRYSTAL_STRUCTURE = "crystal_structure"
    MOLECULAR_STRUCTURE = "molecular_structure"
    TABULAR_DATA = "tabular_data"
    KNOWLEDGE_GRAPH = "knowledge_graph"


# =============================================================================
# Data Models for Benchmark Items
# =============================================================================


@dataclass
class BenchmarkItem(ABC):
    """Base class for all benchmark items."""

    item_id: str
    category: BenchmarkCategory
    difficulty: Difficulty
    description: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def get_query(self) -> Any:
        """Return the query for this benchmark item."""
        pass

    @abstractmethod
    def get_ground_truth(self) -> Any:
        """Return the ground truth for evaluation."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            "category": self.category.value,
            "difficulty": self.difficulty.value,
        }


@dataclass
class CrossScaleRetrievalItem(BenchmarkItem):
    """
    Benchmark item for cross-scale retrieval.

    Tests ability to link information across vastly different scales,
    e.g., DFT bandgap calculation → relevant export control policy.
    """

    source_scale: ScaleLevel = ScaleLevel.ATOMISTIC
    target_scale: ScaleLevel = ScaleLevel.NATIONAL
    source_content: str = ""
    source_content_type: str = "text"
    relevant_target_ids: list[str] = field(default_factory=list)
    relevant_target_contents: list[str] = field(default_factory=list)
    irrelevant_distractors: list[str] = field(default_factory=list)
    bridging_concepts: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.category = BenchmarkCategory.CROSS_SCALE_RETRIEVAL

    def get_query(self) -> str:
        return self.source_content

    def get_ground_truth(self) -> list[str]:
        return self.relevant_target_ids


@dataclass
class CrossModalAlignmentItem(BenchmarkItem):
    """
    Benchmark item for cross-modal alignment.

    Tests ability to retrieve across modalities,
    e.g., XRD spectrum → relevant text documents.
    """

    source_modality: ModalityType = ModalityType.SPECTRUM_XRD
    target_modality: ModalityType = ModalityType.TEXT_SCIENTIFIC
    source_content: Any = None  # Could be spectrum array, structure dict, etc.
    source_file_path: str | None = None
    relevant_target_ids: list[str] = field(default_factory=list)
    relevant_target_contents: list[str] = field(default_factory=list)
    mineral_or_material: str | None = None

    def __post_init__(self):
        self.category = BenchmarkCategory.CROSS_MODAL_ALIGNMENT

    def get_query(self) -> Any:
        return self.source_content or self.source_file_path

    def get_ground_truth(self) -> list[str]:
        return self.relevant_target_ids


@dataclass
class EntityResolutionItem(BenchmarkItem):
    """
    Benchmark item for entity resolution.

    Tests ability to recognize that different mentions refer to the same entity,
    e.g., "TFM" = "Tenke Fungurume" = "Tenke Mining Corp".
    """

    canonical_entity: str = ""
    entity_type: str = ""  # "mine", "company", "mineral", "country", etc.
    aliases: list[str] = field(default_factory=list)
    context_mentions: list[dict[str, str]] = field(
        default_factory=list
    )  # {"context": ..., "mention": ...}
    negative_examples: list[str] = field(default_factory=list)  # Similar but different entities
    wikidata_id: str | None = None
    usgs_id: str | None = None

    def __post_init__(self):
        self.category = BenchmarkCategory.ENTITY_RESOLUTION

    def get_query(self) -> str:
        return self.canonical_entity

    def get_ground_truth(self) -> set[str]:
        return set([self.canonical_entity, *self.aliases])


@dataclass
class SupplyChainTraversalItem(BenchmarkItem):
    """
    Benchmark item for supply chain traversal.

    Tests ability to trace multi-hop paths through the supply chain,
    e.g., "Trace cobalt from DRC mines to US EV batteries".
    """

    query_text: str = ""
    start_entity: str = ""
    start_entity_type: str = ""
    end_entity: str | None = None
    end_entity_type: str | None = None
    expected_path_entities: list[str] = field(default_factory=list)
    expected_path_length: int = 0
    expected_relationships: list[str] = field(default_factory=list)
    alternative_valid_paths: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        self.category = BenchmarkCategory.SUPPLY_CHAIN_TRAVERSAL

    def get_query(self) -> str:
        return self.query_text

    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "primary_path": self.expected_path_entities,
            "relationships": self.expected_relationships,
            "alternatives": self.alternative_valid_paths,
        }


@dataclass
class TemporalConsistencyItem(BenchmarkItem):
    """
    Benchmark item for temporal consistency.

    Tests ability to handle time-sensitive information correctly,
    e.g., distinguishing current sanctions from historical ones.
    """

    query_text: str = ""
    query_timestamp: str = ""  # ISO format timestamp
    time_sensitive_facts: list[dict[str, Any]] = field(default_factory=list)
    # Each fact: {"fact": str, "valid_from": str, "valid_to": str|None, "supersedes": str|None}
    expected_current_answer: str = ""
    expected_historical_answers: dict[str, str] = field(default_factory=dict)  # timestamp -> answer

    def __post_init__(self):
        self.category = BenchmarkCategory.TEMPORAL_CONSISTENCY

    def get_query(self) -> tuple[str, str]:
        return (self.query_text, self.query_timestamp)

    def get_ground_truth(self) -> str:
        return self.expected_current_answer


# =============================================================================
# Benchmark Suite
# =============================================================================


@dataclass
class BenchmarkSuite:
    """
    Complete benchmark suite for CMM embedding evaluation.

    Contains all benchmark items organized by category.
    """

    name: str = "CMM Embedding Benchmark"
    version: str = "1.0.0"
    description: str = "Comprehensive evaluation suite for CMM multi-modal embeddings"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    cross_scale_items: list[CrossScaleRetrievalItem] = field(default_factory=list)
    cross_modal_items: list[CrossModalAlignmentItem] = field(default_factory=list)
    entity_resolution_items: list[EntityResolutionItem] = field(default_factory=list)
    supply_chain_items: list[SupplyChainTraversalItem] = field(default_factory=list)
    temporal_items: list[TemporalConsistencyItem] = field(default_factory=list)

    def get_all_items(self) -> list[BenchmarkItem]:
        """Get all benchmark items across categories."""
        return (
            self.cross_scale_items
            + self.cross_modal_items
            + self.entity_resolution_items
            + self.supply_chain_items
            + self.temporal_items
        )

    def get_items_by_category(self, category: BenchmarkCategory) -> list[BenchmarkItem]:
        """Get items for a specific category."""
        mapping = {
            BenchmarkCategory.CROSS_SCALE_RETRIEVAL: self.cross_scale_items,
            BenchmarkCategory.CROSS_MODAL_ALIGNMENT: self.cross_modal_items,
            BenchmarkCategory.ENTITY_RESOLUTION: self.entity_resolution_items,
            BenchmarkCategory.SUPPLY_CHAIN_TRAVERSAL: self.supply_chain_items,
            BenchmarkCategory.TEMPORAL_CONSISTENCY: self.temporal_items,
        }
        return mapping.get(category, [])

    def get_items_by_difficulty(self, difficulty: Difficulty) -> list[BenchmarkItem]:
        """Get items at a specific difficulty level."""
        return [item for item in self.get_all_items() if item.difficulty == difficulty]

    def get_statistics(self) -> dict[str, Any]:
        """Get benchmark statistics."""
        all_items = self.get_all_items()

        category_counts = {}
        difficulty_counts = {}

        for item in all_items:
            cat = item.category.value
            diff = item.difficulty.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        return {
            "total_items": len(all_items),
            "by_category": category_counts,
            "by_difficulty": difficulty_counts,
            "version": self.version,
        }

    def save(self, path: str):
        """Save benchmark suite to JSON file."""
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "statistics": self.get_statistics(),
            "cross_scale_items": [item.to_dict() for item in self.cross_scale_items],
            "cross_modal_items": [item.to_dict() for item in self.cross_modal_items],
            "entity_resolution_items": [item.to_dict() for item in self.entity_resolution_items],
            "supply_chain_items": [item.to_dict() for item in self.supply_chain_items],
            "temporal_items": [item.to_dict() for item in self.temporal_items],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> BenchmarkSuite:
        """Load benchmark suite from JSON file."""
        with open(path) as f:
            data = json.load(f)

        suite = cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            created_at=data["created_at"],
        )

        # Reconstruct items (simplified - would need proper deserialization)
        # ... implementation details ...

        return suite


# =============================================================================
# Benchmark Item Generators
# =============================================================================


class CrossScaleBenchmarkGenerator:
    """
    Generate cross-scale retrieval benchmark items.

    These test the most challenging aspect of CMM embeddings:
    linking information across vastly different scales.
    """

    @staticmethod
    def generate_atomistic_to_policy_items() -> list[CrossScaleRetrievalItem]:
        """Generate items linking atomistic calculations to policy documents."""
        items = []

        # Item 1: Battery cathode DFT to export control
        items.append(
            CrossScaleRetrievalItem(
                item_id="cs_001",
                difficulty=Difficulty.HARD,
                description="Link LiCoO2 DFT calculation to cobalt export controls",
                source_scale=ScaleLevel.ATOMISTIC,
                target_scale=ScaleLevel.NATIONAL,
                source_content="""
            DFT calculation for LiCoO2 (lithium cobalt oxide):
            - Space group: R-3m (trigonal)
            - Formation energy: -2.31 eV/atom
            - Band gap: 2.1 eV (indirect)
            - Theoretical capacity: 274 mAh/g
            - Intercalation voltage: 3.9 V vs Li/Li+

            This layered oxide is the original lithium-ion battery cathode material,
            with cobalt providing structural stability during lithium extraction.
            """,
                relevant_target_ids=["policy_cobalt_drc_001", "policy_battery_supply_001"],
                relevant_target_contents=[
                    "Section 232 investigation into cobalt imports from Democratic Republic of Congo...",
                    "DOE Battery Materials Strategy emphasizes reducing cobalt dependence...",
                ],
                irrelevant_distractors=[
                    "Agricultural policy on fertilizer imports...",
                    "Steel tariff determinations under Section 201...",
                ],
                bridging_concepts=["lithium-ion battery", "cobalt", "cathode", "supply chain"],
            )
        )

        # Item 2: Rare earth magnet properties to defense policy
        items.append(
            CrossScaleRetrievalItem(
                item_id="cs_002",
                difficulty=Difficulty.EXPERT,
                description="Link NdFeB magnetic properties to defense procurement policy",
                source_scale=ScaleLevel.ATOMISTIC,
                target_scale=ScaleLevel.NATIONAL,
                source_content="""
            Nd2Fe14B (neodymium iron boron) magnetic properties:
            - Crystal structure: Tetragonal P42/mnm
            - Saturation magnetization: 1.28 T
            - Coercivity (intrinsic): 1.2 MA/m
            - Curie temperature: 585 K
            - Maximum energy product (BH)max: 512 kJ/m³

            The high anisotropy field from Nd 4f electrons provides exceptional
            coercivity for permanent magnet applications.
            """,
                relevant_target_ids=["policy_rare_earth_defense_001", "policy_magnet_supply_001"],
                relevant_target_contents=[
                    "Defense Production Act Title III determination for rare earth permanent magnets...",
                    "National Defense Stockpile requirements for neodymium and dysprosium...",
                ],
                irrelevant_distractors=[
                    "Consumer electronics recycling regulations...",
                    "Automotive fuel efficiency standards...",
                ],
                bridging_concepts=[
                    "permanent magnet",
                    "rare earth",
                    "neodymium",
                    "defense",
                    "F-35",
                ],
            )
        )

        # Item 3: Semiconductor bandgap to CHIPS Act
        items.append(
            CrossScaleRetrievalItem(
                item_id="cs_003",
                difficulty=Difficulty.HARD,
                description="Link GaN semiconductor properties to CHIPS Act provisions",
                source_scale=ScaleLevel.ATOMISTIC,
                target_scale=ScaleLevel.NATIONAL,
                source_content="""
            Gallium Nitride (GaN) electronic properties:
            - Band gap: 3.4 eV (direct)
            - Electron mobility: 1000-2000 cm²/Vs
            - Breakdown field: 3.3 MV/cm
            - Thermal conductivity: 130 W/mK
            - Saturation velocity: 2.5 × 10⁷ cm/s

            Wide bandgap enables high-power, high-frequency operation
            for 5G infrastructure and electric vehicle inverters.
            """,
                relevant_target_ids=["policy_chips_act_001", "policy_gallium_001"],
                relevant_target_contents=[
                    "CHIPS and Science Act Section 103 funding for wide bandgap semiconductor R&D...",
                    "Commerce Department restrictions on gallium exports to adversarial nations...",
                ],
                irrelevant_distractors=[
                    "Solar panel manufacturing incentives...",
                    "Telecommunications spectrum auction rules...",
                ],
                bridging_concepts=[
                    "gallium",
                    "semiconductor",
                    "5G",
                    "power electronics",
                    "wide bandgap",
                ],
            )
        )

        return items

    @staticmethod
    def generate_material_to_trade_items() -> list[CrossScaleRetrievalItem]:
        """Generate items linking material properties to trade data."""
        items = []

        # Item 4: Lithium brine chemistry to trade flows
        items.append(
            CrossScaleRetrievalItem(
                item_id="cs_004",
                difficulty=Difficulty.MEDIUM,
                description="Link lithium extraction chemistry to Chile-China trade",
                source_scale=ScaleLevel.MATERIAL,
                target_scale=ScaleLevel.GLOBAL,
                source_content="""
            Lithium extraction from Salar de Atacama brine:
            - Li concentration: 1,500-3,000 ppm
            - Mg/Li ratio: 6.4 (favorable for extraction)
            - Evaporation time: 12-18 months
            - Recovery rate: 50-70%
            - Final product: Li2CO3 (battery grade >99.5%)

            Solar evaporation followed by lime precipitation removes Mg,
            then soda ash precipitates lithium carbonate.
            """,
                relevant_target_ids=["trade_chile_china_li_001", "trade_lithium_price_001"],
                relevant_target_contents=[
                    "Chile lithium carbonate exports to China reached 85,000 tonnes in 2024...",
                    "Lithium carbonate spot prices in Shanghai increased 40% following...",
                ],
                bridging_concepts=[
                    "lithium carbonate",
                    "Chile",
                    "brine",
                    "battery grade",
                    "export",
                ],
            )
        )

        return items


class CrossModalBenchmarkGenerator:
    """
    Generate cross-modal alignment benchmark items.

    Tests retrieval across different data modalities.
    """

    @staticmethod
    def generate_spectrum_to_text_items() -> list[CrossModalAlignmentItem]:
        """Generate items for spectrum → text retrieval."""
        items = []

        # Item 1: Cobaltite XRD to geological text
        items.append(
            CrossModalAlignmentItem(
                item_id="cm_001",
                difficulty=Difficulty.MEDIUM,
                description="Match cobaltite XRD pattern to geological survey text",
                source_modality=ModalityType.SPECTRUM_XRD,
                target_modality=ModalityType.TEXT_SCIENTIFIC,
                source_content={
                    "mineral": "cobaltite",
                    "formula": "CoAsS",
                    "peaks_2theta": [28.5, 32.1, 36.2, 46.8, 52.3, 58.7],
                    "peak_intensities": [100, 45, 78, 23, 35, 12],
                    "crystal_system": "orthorhombic",
                },
                source_file_path="data/spectra/cobaltite_001.xy",
                relevant_target_ids=["text_cobaltite_geology_001", "text_drc_minerals_001"],
                relevant_target_contents=[
                    "Cobaltite (CoAsS) occurs in hydrothermal vein deposits associated with...",
                    "The Katanga Copperbelt hosts significant cobaltite mineralization...",
                ],
                mineral_or_material="cobaltite",
            )
        )

        # Item 2: Coltan XRF to supply chain text
        items.append(
            CrossModalAlignmentItem(
                item_id="cm_002",
                difficulty=Difficulty.HARD,
                description="Match coltan XRF fingerprint to sourcing documentation",
                source_modality=ModalityType.SPECTRUM_XRF,
                target_modality=ModalityType.TEXT_POLICY,
                source_content={
                    "mineral": "columbite-tantalite",
                    "elements": {
                        "Ta": 32.5,  # weight %
                        "Nb": 28.3,
                        "Fe": 12.1,
                        "Mn": 8.7,
                        "Sn": 2.1,
                    },
                    "origin_signature": "DRC_Kivu",
                },
                relevant_target_ids=["text_conflict_minerals_001", "text_dodd_frank_1502"],
                relevant_target_contents=[
                    "Dodd-Frank Section 1502 requires disclosure of conflict mineral sourcing...",
                    "ITSCI tagging system for tantalum ore from certified DRC mines...",
                ],
                mineral_or_material="coltan",
            )
        )

        return items

    @staticmethod
    def generate_structure_to_text_items() -> list[CrossModalAlignmentItem]:
        """Generate items for crystal structure → text retrieval."""
        items = []

        # Item 1: Spodumene structure to lithium mining text
        items.append(
            CrossModalAlignmentItem(
                item_id="cm_003",
                difficulty=Difficulty.MEDIUM,
                description="Match spodumene crystal structure to mining text",
                source_modality=ModalityType.CRYSTAL_STRUCTURE,
                target_modality=ModalityType.TEXT_SCIENTIFIC,
                source_content={
                    "formula": "LiAlSi2O6",
                    "crystal_system": "monoclinic",
                    "space_group": "C2/c",
                    "lattice_parameters": {
                        "a": 9.46,
                        "b": 8.39,
                        "c": 5.22,
                        "beta": 110.5,
                    },
                    "density": 3.18,  # g/cm³
                },
                relevant_target_ids=["text_spodumene_processing_001", "text_hard_rock_li_001"],
                relevant_target_contents=[
                    "Spodumene processing requires calcination at 1050°C to convert α to β phase...",
                    "Western Australian hard rock lithium mines produce spodumene concentrate...",
                ],
                mineral_or_material="spodumene",
            )
        )

        return items


class EntityResolutionBenchmarkGenerator:
    """
    Generate entity resolution benchmark items.

    Tests ability to match entities across naming variations.
    """

    @staticmethod
    def generate_mine_entity_items() -> list[EntityResolutionItem]:
        """Generate entity resolution items for mines."""
        items = []

        # Tenke Fungurume Mine
        items.append(
            EntityResolutionItem(
                item_id="er_001",
                difficulty=Difficulty.MEDIUM,
                description="Resolve Tenke Fungurume Mine name variations",
                canonical_entity="Tenke Fungurume Mine",
                entity_type="mine",
                aliases=[
                    "TFM",
                    "Tenke-Fungurume",
                    "Tenke Fungurume Mining",
                    "Tenke Mining Corp",
                    "TF Mine",
                    "腾科丰谷鲁米矿",  # Chinese
                ],
                context_mentions=[
                    {"context": "CMOC's TFM operation in Katanga province...", "mention": "TFM"},
                    {
                        "context": "Cobalt production at Tenke-Fungurume exceeded...",
                        "mention": "Tenke-Fungurume",
                    },
                    {
                        "context": "The Tenke Fungurume Mining complex includes...",
                        "mention": "Tenke Fungurume Mining",
                    },
                ],
                negative_examples=[
                    "Mutanda Mining",  # Different mine, similar region
                    "Kamoto Mine",  # Different mine
                    "Fungurume Township",  # Location, not mine
                ],
                wikidata_id="Q2399394",
            )
        )

        # Escondida Mine
        items.append(
            EntityResolutionItem(
                item_id="er_002",
                difficulty=Difficulty.EASY,
                description="Resolve Escondida Mine name variations",
                canonical_entity="Escondida Mine",
                entity_type="mine",
                aliases=[
                    "Minera Escondida",
                    "Escondida copper mine",
                    "BHP Escondida",
                    "MEL",  # Minera Escondida Limitada
                ],
                context_mentions=[
                    {"context": "BHP's Escondida operations in Chile...", "mention": "Escondida"},
                    {
                        "context": "Minera Escondida Limitada reported...",
                        "mention": "Minera Escondida Limitada",
                    },
                ],
                negative_examples=[
                    "Spence Mine",
                    "Collahuasi",
                ],
                wikidata_id="Q1350844",
            )
        )

        return items

    @staticmethod
    def generate_company_entity_items() -> list[EntityResolutionItem]:
        """Generate entity resolution items for companies."""
        items = []

        # CMOC Group
        items.append(
            EntityResolutionItem(
                item_id="er_003",
                difficulty=Difficulty.HARD,
                description="Resolve CMOC Group name variations and subsidiaries",
                canonical_entity="CMOC Group Limited",
                entity_type="company",
                aliases=[
                    "CMOC",
                    "China Molybdenum Co",
                    "China Moly",
                    "洛阳钼业",  # Chinese name
                    "Luoyang Molybdenum",
                    "603993.SS",  # Stock ticker
                    "3993.HK",
                ],
                context_mentions=[
                    {"context": "CMOC acquired Freeport's DRC assets...", "mention": "CMOC"},
                    {
                        "context": "China Molybdenum Co., Ltd. reported...",
                        "mention": "China Molybdenum Co., Ltd.",
                    },
                    {"context": "洛阳钼业集团的钴产量...", "mention": "洛阳钼业集团"},
                ],
                negative_examples=[
                    "China Northern Rare Earth",
                    "Jinchuan Group",
                    "MMG Limited",
                ],
                wikidata_id="Q10873127",
            )
        )

        # Glencore
        items.append(
            EntityResolutionItem(
                item_id="er_004",
                difficulty=Difficulty.MEDIUM,
                description="Resolve Glencore name variations",
                canonical_entity="Glencore plc",
                entity_type="company",
                aliases=[
                    "Glencore",
                    "Glencore International",
                    "Glencore Xstrata",  # Historical
                    "GLEN.L",
                    "嘉能可",  # Chinese
                ],
                context_mentions=[
                    {"context": "Glencore's Katanga Mining subsidiary...", "mention": "Glencore"},
                    {
                        "context": "Following the Glencore Xstrata merger...",
                        "mention": "Glencore Xstrata",
                    },
                ],
                negative_examples=[
                    "Trafigura",
                    "Vitol",
                    "Mercuria",
                ],
                wikidata_id="Q929278",
            )
        )

        return items

    @staticmethod
    def generate_mineral_entity_items() -> list[EntityResolutionItem]:
        """Generate entity resolution items for minerals/materials."""
        items = []

        # Rare earth elements
        items.append(
            EntityResolutionItem(
                item_id="er_005",
                difficulty=Difficulty.MEDIUM,
                description="Resolve rare earth element naming",
                canonical_entity="Neodymium",
                entity_type="element",
                aliases=[
                    "Nd",
                    "neodymium",
                    "neo",  # Industry shorthand
                    "钕",  # Chinese
                ],
                context_mentions=[
                    {"context": "Nd content in the ore averaged...", "mention": "Nd"},
                    {
                        "context": "Neodymium oxide prices increased...",
                        "mention": "Neodymium oxide",
                    },
                    {"context": "Neo-rich rare earth concentrate...", "mention": "Neo"},
                ],
                negative_examples=[
                    "Praseodymium",
                    "Samarium",
                    "NdFeB",  # Compound, not element
                ],
                wikidata_id="Q1388",
            )
        )

        return items


class SupplyChainBenchmarkGenerator:
    """
    Generate supply chain traversal benchmark items.

    Tests multi-hop reasoning through supply chain graph.
    """

    @staticmethod
    def generate_items() -> list[SupplyChainTraversalItem]:
        """Generate supply chain traversal items."""
        items = []

        # Cobalt: DRC mine to US EV
        items.append(
            SupplyChainTraversalItem(
                item_id="sc_001",
                difficulty=Difficulty.MEDIUM,
                description="Trace cobalt from DRC mine to US EV battery",
                query_text="Trace the supply chain for cobalt from Tenke Fungurume mine to Tesla vehicles",
                start_entity="Tenke Fungurume Mine",
                start_entity_type="mine",
                end_entity="Tesla Model 3",
                end_entity_type="product",
                expected_path_entities=[
                    "Tenke Fungurume Mine",
                    "CMOC Group",
                    "Huayou Cobalt",
                    "CATL",
                    "Tesla Gigafactory",
                    "Tesla Model 3",
                ],
                expected_path_length=6,
                expected_relationships=[
                    "OPERATED_BY",
                    "SUPPLIES_TO",
                    "PROCESSES_FOR",
                    "SUPPLIES_TO",
                    "MANUFACTURES",
                ],
                alternative_valid_paths=[
                    [
                        "Tenke Fungurume Mine",
                        "CMOC Group",
                        "Umicore",
                        "Samsung SDI",
                        "Ford",
                        "Ford Mustang Mach-E",
                    ],
                ],
            )
        )

        # Rare earth: China to US defense
        items.append(
            SupplyChainTraversalItem(
                item_id="sc_002",
                difficulty=Difficulty.HARD,
                description="Trace rare earth elements to F-35 magnets",
                query_text="Identify the rare earth supply chain for F-35 permanent magnets",
                start_entity="Bayan Obo Mine",
                start_entity_type="mine",
                end_entity="F-35 Lightning II",
                end_entity_type="product",
                expected_path_entities=[
                    "Bayan Obo Mine",
                    "China Northern Rare Earth",
                    "Baotou Processing",
                    "Shin-Etsu Chemical",
                    "Lockheed Martin",
                    "F-35 Lightning II",
                ],
                expected_path_length=6,
                expected_relationships=[
                    "OPERATED_BY",
                    "PROCESSES",
                    "SUPPLIES_TO",
                    "SUPPLIES_TO",
                    "MANUFACTURES",
                ],
            )
        )

        # Lithium: Australia to battery cell
        items.append(
            SupplyChainTraversalItem(
                item_id="sc_003",
                difficulty=Difficulty.EASY,
                description="Trace lithium from Australian mine to battery",
                query_text="How does lithium from Greenbushes reach battery production?",
                start_entity="Greenbushes Mine",
                start_entity_type="mine",
                end_entity="Lithium-ion battery cell",
                end_entity_type="product",
                expected_path_entities=[
                    "Greenbushes Mine",
                    "Tianqi Lithium",
                    "Kwinana Refinery",
                    "LG Energy Solution",
                    "Lithium-ion battery cell",
                ],
                expected_path_length=5,
                expected_relationships=[
                    "JV_OPERATED_BY",
                    "REFINES_AT",
                    "SUPPLIES_TO",
                    "MANUFACTURES",
                ],
            )
        )

        return items


class TemporalBenchmarkGenerator:
    """
    Generate temporal consistency benchmark items.

    Tests handling of time-sensitive information.
    """

    @staticmethod
    def generate_items() -> list[TemporalConsistencyItem]:
        """Generate temporal consistency items."""
        items = []

        # Sanctions evolution
        items.append(
            TemporalConsistencyItem(
                item_id="tc_001",
                difficulty=Difficulty.HARD,
                description="Track Russian aluminum sanctions over time",
                query_text="What are the US sanctions on Russian aluminum imports?",
                query_timestamp="2024-06-01T00:00:00Z",
                time_sensitive_facts=[
                    {
                        "fact": "Section 232 tariffs of 200% on Russian aluminum",
                        "valid_from": "2024-03-01",
                        "valid_to": None,
                        "supersedes": "pre_2024_sanctions",
                    },
                    {
                        "fact": "10% tariff on Russian aluminum under Section 232",
                        "valid_from": "2022-03-01",
                        "valid_to": "2024-02-28",
                        "supersedes": "pre_2022_tariffs",
                    },
                    {
                        "fact": "No specific sanctions on Russian aluminum",
                        "valid_from": "2018-01-01",
                        "valid_to": "2022-02-28",
                        "supersedes": None,
                    },
                ],
                expected_current_answer="200% tariff under Section 232 (effective March 2024)",
                expected_historical_answers={
                    "2023-06-01": "10% tariff under Section 232",
                    "2021-06-01": "No specific sanctions",
                },
            )
        )

        # Export control changes
        items.append(
            TemporalConsistencyItem(
                item_id="tc_002",
                difficulty=Difficulty.MEDIUM,
                description="Track China gallium export controls",
                query_text="What are China's export restrictions on gallium?",
                query_timestamp="2024-01-01T00:00:00Z",
                time_sensitive_facts=[
                    {
                        "fact": "Export license required for gallium and germanium",
                        "valid_from": "2023-08-01",
                        "valid_to": None,
                        "supersedes": "pre_2023_no_restriction",
                    },
                    {
                        "fact": "No export restrictions on gallium",
                        "valid_from": "2000-01-01",
                        "valid_to": "2023-07-31",
                        "supersedes": None,
                    },
                ],
                expected_current_answer="Export license required since August 2023",
                expected_historical_answers={
                    "2023-01-01": "No export restrictions",
                },
            )
        )

        return items


# =============================================================================
# Benchmark Suite Factory
# =============================================================================


def create_full_benchmark_suite() -> BenchmarkSuite:
    """
    Create the complete CMM benchmark suite with all categories.

    Returns:
        BenchmarkSuite with all benchmark items
    """
    suite = BenchmarkSuite(
        name="CMM Embedding Benchmark v1.0",
        version="1.0.0",
        description="Comprehensive evaluation suite for CMM multi-modal embeddings",
    )

    # Cross-scale retrieval
    suite.cross_scale_items.extend(
        CrossScaleBenchmarkGenerator.generate_atomistic_to_policy_items()
    )
    suite.cross_scale_items.extend(CrossScaleBenchmarkGenerator.generate_material_to_trade_items())

    # Cross-modal alignment
    suite.cross_modal_items.extend(CrossModalBenchmarkGenerator.generate_spectrum_to_text_items())
    suite.cross_modal_items.extend(CrossModalBenchmarkGenerator.generate_structure_to_text_items())

    # Entity resolution
    suite.entity_resolution_items.extend(
        EntityResolutionBenchmarkGenerator.generate_mine_entity_items()
    )
    suite.entity_resolution_items.extend(
        EntityResolutionBenchmarkGenerator.generate_company_entity_items()
    )
    suite.entity_resolution_items.extend(
        EntityResolutionBenchmarkGenerator.generate_mineral_entity_items()
    )

    # Supply chain traversal
    suite.supply_chain_items.extend(SupplyChainBenchmarkGenerator.generate_items())

    # Temporal consistency
    suite.temporal_items.extend(TemporalBenchmarkGenerator.generate_items())

    return suite


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CMM benchmark suite")
    parser.add_argument("--output", "-o", default="cmm_benchmark_v1.json")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")

    args = parser.parse_args()

    suite = create_full_benchmark_suite()

    if args.stats:
        import pprint

        pprint.pprint(suite.get_statistics())
    else:
        suite.save(args.output)
        print(f"Saved benchmark suite to {args.output}")
        print(f"Statistics: {suite.get_statistics()}")
