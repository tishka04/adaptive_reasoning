"""Learning suite for V4."""

from __future__ import annotations

from dataclasses import dataclass, field

from .arbiter_bandit import ArbiterBandit
from .bridge_value_model import BridgeValueModel
from .compression_value_model import CompressionValueModel
from .ontology_calibrator import OntologyCalibrator
from .operator_utility_model import OperatorUtilityModel
from .project_value_model import ProjectValueModel
from .retrospective_credit import RetrospectiveCredit
from .sterility_predictor import SterilityPredictor
from .teleology_validator import TeleologyValidator
from .world_embedding_model import WorldEmbeddingModel
from .world_reliability_model import WorldReliabilityModel


@dataclass
class LearningSuite:
    """Bundle the lightweight online learning components for V4."""

    world_reliability: WorldReliabilityModel = field(default_factory=WorldReliabilityModel)
    project_value: ProjectValueModel = field(default_factory=ProjectValueModel)
    arbiter_bandit: ArbiterBandit = field(default_factory=ArbiterBandit)
    ontology_calibrator: OntologyCalibrator = field(default_factory=OntologyCalibrator)
    operator_utility: OperatorUtilityModel = field(default_factory=OperatorUtilityModel)
    teleology_validator: TeleologyValidator = field(default_factory=TeleologyValidator)
    sterility_predictor: SterilityPredictor = field(default_factory=SterilityPredictor)
    bridge_value: BridgeValueModel = field(default_factory=BridgeValueModel)
    compression_value: CompressionValueModel = field(default_factory=CompressionValueModel)
    world_embedding: WorldEmbeddingModel = field(default_factory=WorldEmbeddingModel)
    credit: RetrospectiveCredit = field(init=False)

    def __post_init__(self) -> None:
        self.credit = RetrospectiveCredit(self)

    def seed_from_cross_game(self, cross_game) -> None:
        self.world_reliability.seed_priors(getattr(cross_game, "learned_ontology_priors", {}))
        self.project_value.seed_priors(getattr(cross_game, "learned_project_priors", {}))
        self.ontology_calibrator.seed_priors(getattr(cross_game, "learned_ontology_priors", {}))
        self.bridge_value.seed_priors(getattr(cross_game, "learned_bridge_priors", {}))
        self.compression_value.seed_priors(getattr(cross_game, "learned_compression_priors", {}))
        self.world_embedding.seed_from_cross_game(cross_game)

    def export_to_cross_game(self, cross_game) -> None:
        cross_game.learned_ontology_priors = self.ontology_calibrator.export_priors()
        cross_game.learned_project_priors = self.project_value.export_priors()
        cross_game.learned_bridge_priors = self.bridge_value.export_priors()
        cross_game.learned_compression_priors = self.compression_value.export_priors()
        self.world_embedding.export_to_cross_game(cross_game)
