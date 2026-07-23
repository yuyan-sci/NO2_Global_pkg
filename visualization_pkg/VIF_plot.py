from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def build_X_from_ds(ds, total_channel_names):
    # use center of any odd-sized local grid (5x5 -> 2, 11x11 -> 5)
    grid_dims = [d for d in ds.dims if ds.dims[d] % 2 == 1 and ds.dims[d] <= 11]
    sel = {d: ds.dims[d] // 2 for d in grid_dims}

    cols, lengths = {}, []
    for name in total_channel_names:
        if name in ds:
            v = ds[name].isel(**sel).values.reshape(-1)  # flatten over remaining dims
            cols[name] = v
            lengths.append(len(v))
        else:
            cols[name] = None

    # keep the most common length across variables
    common_len = max(set(lengths), key=lengths.count)
    X = np.column_stack([
        (cols[name] if (cols[name] is not None and len(cols[name]) == common_len)
         else np.full(common_len, np.nan))
        for name in total_channel_names
    ])
    return X

def _vif_corrinv(Xm):
    Xm = np.asarray(Xm, float)
    # mean-impute NaNs
    col_mean = np.nanmean(Xm, axis=0)
    inds = np.where(np.isnan(Xm))
    if inds[0].size:
        Xm[inds] = np.take(col_mean, inds[1])
    # standardize
    mu = Xm.mean(axis=0); sd = Xm.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (Xm - mu) / sd
    # corr inverse -> diag(inv(corr)) = VIF
    n = Z.shape[0]
    corr = (Z.T @ Z) / max(n - 1, 1)
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr, rcond=1e-8)
    vif = np.diag(inv_corr).astype(float)
    return np.clip(vif, 1.0, 1e9)

def _vif_statsmodels(Xm):
    """
    Xm: (n_samples, p) design matrix (no target), may contain NaNs/constant cols.
    Returns VIF length p (NaN for dropped-constant cols).
    """
    Xm = np.asarray(Xm, float)

    # mean-impute NaNs
    col_mean = np.nanmean(Xm, axis=0)
    inds = np.where(np.isnan(Xm))
    if inds[0].size:
        Xm[inds] = np.take(col_mean, inds[1])

    # drop constant columns (statsmodels VIF fails on zero variance)
    std = Xm.std(axis=0, ddof=1)
    keep = std > 0
    vif_all = np.full(Xm.shape[1], np.nan, dtype=float)

    # if p >= n, OLS-based VIF is unstable -> fall back
    if Xm.shape[0] <= keep.sum():
        return _vif_corrinv(Xm)

    Xm_keep = Xm[:, keep]
    vals = [variance_inflation_factor(Xm_keep, i) for i in range(Xm_keep.shape[1])]
    vif_all[keep] = np.array(vals, float)
    return np.clip(vif_all, 1.0, 1e9)