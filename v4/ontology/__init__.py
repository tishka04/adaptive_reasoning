"""Ontology chamber for V4."""

from .affordance_reframing import AffordanceReframer
from .ontology_competition import OntologyCompetition
from .ontology_hypotheses import ONTOLOGY_KINDS, OntologyFactory

__all__ = [
    "AffordanceReframer",
    "ONTOLOGY_KINDS",
    "OntologyCompetition",
    "OntologyFactory",
]
