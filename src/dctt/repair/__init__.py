"""Token embedding repair methods."""

from dctt.repair.losses import (
    GeometryLoss,
    AnchorLoss,
    NeighborPreservationLoss,
    CombinedRepairLoss,
)
from dctt.repair.optimizer import (
    EmbeddingRepairOptimizer,
    repair_embedding,
    repair_embeddings_batch,
)
from dctt.repair.projection import orthogonalize_embedding, ProjectionRepair
from dctt.repair.candidate import select_repair_candidates, CandidateSelector
from dctt.repair.validate import (
    validate_semantic_preservation,
    SemanticValidator,
)

__all__ = [
    # Losses
    "GeometryLoss",
    "AnchorLoss",
    "NeighborPreservationLoss",
    "CombinedRepairLoss",
    # Optimizer
    "EmbeddingRepairOptimizer",
    "repair_embedding",
    "repair_embeddings_batch",
    # Projection
    "orthogonalize_embedding",
    "ProjectionRepair",
    # Candidates
    "select_repair_candidates",
    "CandidateSelector",
    # Validation
    "validate_semantic_preservation",
    "SemanticValidator",
]
