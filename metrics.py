import numpy as np


def mae(preds, gts):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    gts = np.asarray(gts, dtype=np.float64).reshape(-1)
    if preds.shape != gts.shape:
        raise ValueError(f"shape mismatch: preds={preds.shape}, gts={gts.shape}")
    return float(np.mean(np.abs(preds - gts)))


def forgetting_matrix_stats(M):
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square task x task matrix")
    T = M.shape[0]
    final_seen = M[T - 1, :T]
    final_avg = float(np.nanmean(final_seen))
    oldest = float(M[T - 1, 0])
    newest = float(M[T - 1, T - 1])

    bwts = []
    for i in range(T - 1):
        if not np.isnan(M[i, i]) and not np.isnan(M[T - 1, i]):
            denom = max(abs(float(M[i, i])), 1e-12)
            bwts.append((float(M[i, i]) - float(M[T - 1, i])) / denom)
    nbwt = float(np.mean(bwts)) if bwts else 0.0

    return {
        "final_avg_mae": final_avg,
        "nBwT": nbwt,
        "final_oldest_mae": oldest,
        "final_newest_mae": newest,
    }

