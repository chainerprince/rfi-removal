import numpy as np


def score_mask(mask, labels, *, valid=None):
    """Compare Boolean detector flags with Boolean injection labels."""
    if mask.dtype != np.bool_ or labels.dtype != np.bool_:
        raise ValueError("mask and labels must be Boolean arrays")
    if mask.shape != labels.shape:
        raise ValueError("mask and labels must have the same shape")
    if valid is None:
        valid = np.ones(labels.shape, bool)
    if valid.shape != labels.shape or valid.dtype != np.bool_:
        raise ValueError("valid must be a Boolean array with the same shape")
    tp = int(np.count_nonzero(mask & labels & valid))
    fp = int(np.count_nonzero(mask & ~labels & valid))
    fn = int(np.count_nonzero(~mask & labels & valid))
    tn = int(np.count_nonzero(~mask & ~labels & valid))
    ratio = lambda a, b: a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                false_positive_rate=ratio(fp, fp+tn), removed=float(mask.mean()),
                unscored=int(np.count_nonzero(~valid)))


def recall_by_inr(mask, inr, edges):
    """Recall in positive cell-INR bins [left, right); empty bins return None."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.asarray(edges, dtype=float)
    if (mask.shape != inr.shape or not np.isfinite(inr).all() or np.any(inr < 0)
            or edges.ndim != 1 or len(edges) < 2 or np.isnan(edges).any()
            or edges[0] <= 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("invalid mask, INR or positive bin edges")
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        positive = (inr >= left) & (inr < right)
        total = int(positive.sum())
        detected = int(np.count_nonzero(mask & positive))
        rows.append(dict(lower=float(left), upper=float(right), positives=total,
                         detected=detected, recall=detected / total if total else None))
    return rows


def score(mask, inr, min_inr=0.01):
    """Score known synthetic truth. Exclude 0 < INR < min_inr from confusion counts."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != inr.shape or min_inr <= 0 or not np.isfinite(inr).all() or np.any(inr < 0):
        raise ValueError("invalid mask shape or INR")
    positive, negative = inr >= min_inr, inr == 0
    tp = int(np.count_nonzero(mask & positive))
    fp = int(np.count_nonzero(mask & negative))
    fn = int(np.count_nonzero(~mask & positive))
    tn = int(np.count_nonzero(~mask & negative))
    ratio = lambda a, b: a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                false_positive_rate=ratio(fp, fp + tn), removed=float(mask.mean()),
                unscored=int(np.count_nonzero(~(positive | negative))))
