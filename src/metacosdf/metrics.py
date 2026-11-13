"""How well a field matches the ground-truth C-space SDF (the metrics of Co-SDF, Table I)."""

from scipy import stats
from torch import Tensor

from metacosdf.tensor_utils import to_numpy


def field_metrics(prediction: Tensor, ground_truth: Tensor) -> dict[str, float]:
    """
    Compare predicted values with ground-truth C-space distances at the same configurations,
    e.g. both evaluated on cspace_grid(res).

    spearman, pearson: does the field rank and scale like the distance to collision?
    accuracy: share of configurations with the right sign.
    fnr: share of colliding configurations predicted free.
    fpr: share of free configurations predicted colliding.
    """
    predicted = to_numpy(prediction).ravel()
    actual = to_numpy(ground_truth).ravel()
    collision, predicted_collision = actual < 0, predicted < 0
    n_collision, n_free = max(collision.sum(), 1), max((~collision).sum(), 1)
    return {
        "spearman": float(stats.spearmanr(predicted, actual).statistic),
        "pearson": float(stats.pearsonr(predicted, actual).statistic),
        "accuracy": float((collision == predicted_collision).mean()),
        "fnr": float((collision & ~predicted_collision).sum() / n_collision),
        "fpr": float((~collision & predicted_collision).sum() / n_free),
    }
