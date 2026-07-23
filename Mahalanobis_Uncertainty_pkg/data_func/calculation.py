import numpy as np

def calculate_covariance_matrix(data):
    """
    Calculate the covariance matrix of the given data.

    Parameters:
    data (numpy.ndarray): A 2D array where each row (axis=0) represents a sample and each column (axis=1) represents a feature.

    Returns:
    numpy.ndarray: The covariance matrix of the data.
    """
    # Center the data by subtracting the mean
    centered_data = data - np.mean(data, axis=0)
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    return covariance_matrix

import numpy as np

def calculate_mahalanobis_distance(data, mean_vector, inverted_covariance_matrix):
    """
    Mahalanobis distance for:
      1) Grid data shaped (n_features, d1, d2): returns (d1, d2)
      2) Tabular data shaped (n_samples, n_features): returns (n_samples,)

    Parameters
    ----------
    data : array_like
        - (n, d1, d2): For each grid cell (d1, d2), an n-dimensional feature vector.
        - (m, f): m samples with f features each.
    mean_vector : array_like
        - For (n, d1, d2) data: (n,), (n, d1, d2), or broadcastable like (n, 1, 1).
        - For (m, f) data: (f,) or (m, f) (per-sample means).
    inverted_covariance_matrix : array_like
        - For (n, d1, d2) data: (n, n) shared or (n, n, d1, d2) per-pixel (broadcastable like (n, n, 1, 1)).
        - For (m, f) data: (f, f) shared or (m, f, f) per-sample.

    Returns
    -------
    distances : ndarray
        - (d1, d2) for grid input.
        - (m,) for tabular input.

    Notes
    -----
    - For per-pixel/per-sample inverse covariances, ensure symmetry/PD.
    - Small negative values from round-off are clamped to zero before sqrt.
    """
    X = np.asarray(data)
    mu = np.asarray(mean_vector)
    inv = np.asarray(inverted_covariance_matrix)

    # -----------------------------
    # Case A: Grid data (n, d1, d2)
    # -----------------------------
    if X.ndim == 3:
        n, d1, d2 = X.shape

        # Mean handling
        if mu.ndim == 1:
            if mu.shape[0] != n:
                raise ValueError(f"Mean length {mu.shape[0]} != feature dim {n}.")
            mu = mu[:, None, None]  # (n,1,1)
        elif mu.ndim == 3:
            if mu.shape[0] != n:
                raise ValueError(f"Mean first dim {mu.shape[0]} != feature dim {n}.")
        else:
            raise ValueError("For (n,d1,d2) data, mean_vector must be (n,) or (n,d1,d2) (broadcastable).")

        try:
            mu = np.broadcast_to(mu, (n, d1, d2))
        except ValueError:
            raise ValueError(f"`mean_vector` not broadcastable to (n,d1,d2)=({n},{d1},{d2}); got {mu.shape}.")

        diff = X - mu  # (n, d1, d2)

        # Inverse covariance
        if inv.ndim == 2:
            if inv.shape != (n, n):
                raise ValueError(f"Inverse covariance must be (n,n)=({n},{n}); got {inv.shape}.")
            left = np.einsum('ij,jhw->ihw', inv, diff, optimize=True)   # (n, d1, d2)
            dist2 = np.einsum('ihw,ihw->hw', left, diff, optimize=True) # (d1, d2)
        elif inv.ndim == 4:
            if inv.shape[0] != n or inv.shape[1] != n:
                raise ValueError(f"Per-pixel inverse covariance first two dims must be (n,n)=({n},{n}); got {inv.shape[:2]}.")
            try:
                inv = np.broadcast_to(inv, (n, n, d1, d2))
            except ValueError:
                raise ValueError(f"`inverted_covariance_matrix` not broadcastable to (n,n,d1,d2)=({n},{n},{d1},{d2}); got {inv.shape}.")
            left = np.einsum('ijhw,jhw->ihw', inv, diff, optimize=True)  # (n, d1, d2)
            dist2 = np.einsum('ihw,ihw->hw', left, diff, optimize=True)  # (d1, d2)
        else:
            raise ValueError("For (n,d1,d2) data, inverse must be (n,n) or (n,n,d1,d2) (broadcastable).")

        return np.sqrt(np.maximum(dist2, 0.0))

    # --------------------------------
    # Case B: Tabular data (m, f)
    # --------------------------------
    elif X.ndim == 2:
        m, f = X.shape

        # Mean handling
        if mu.ndim == 1:
            if mu.shape[0] != f:
                raise ValueError(f"Mean length {mu.shape[0]} != feature dim {f}.")
            mu = np.broadcast_to(mu, (m, f))
        elif mu.ndim == 2:
            if mu.shape != (m, f):
                raise ValueError(f"Per-sample mean must be (m,f)=({m},{f}); got {mu.shape}.")
        else:
            raise ValueError("For (m,f) data, mean_vector must be (f,) or (m,f).")

        diff = X - mu  # (m, f)

        # Inverse covariance
        if inv.ndim == 2:
            if inv.shape != (f, f):
                raise ValueError(f"Inverse covariance must be (f,f)=({f},{f}); got {inv.shape}.")
            # left = diff @ inv^T  (since inv acts on feature dimension)
            left = np.einsum('ij,mj->mi', inv, diff, optimize=True)       # (m, f)
            dist2 = np.einsum('mi,mi->m', left, diff, optimize=True)      # (m,)
        elif inv.ndim == 3:
            if inv.shape != (m, f, f):
                raise ValueError(f"Per-sample inverse covariance must be (m,f,f)=({m},{f},{f}); got {inv.shape}.")
            left = np.einsum('mij,mj->mi', inv, diff, optimize=True)      # (m, f)
            dist2 = np.einsum('mi,mi->m', left, diff, optimize=True)      # (m,)
        else:
            raise ValueError("For (m,f) data, inverse must be (f,f) or (m,f,f).")

        return np.sqrt(np.maximum(dist2, 0.0))

    else:
        raise ValueError(f"`data` must be 2D (m,f) or 3D (n,d1,d2); got shape {X.shape}.")


def invert_matrix(matrix):
    """
    Invert a given square matrix.

    Parameters:
    matrix (numpy.ndarray): A square matrix to be inverted.

    Returns:
    numpy.ndarray: The inverted matrix.
    """
    try:
        inverted_matrix = np.linalg.inv(matrix)
        return inverted_matrix
    except np.linalg.LinAlgError:
        # Handle singular matrix case
        print(matrix.shape)
        ## find where the matrix is singular
        singular_values = np.where(np.isclose(np.linalg.svd(matrix, compute_uv=False), 0))[0]
        print("Singular values found at indices:", singular_values)
        raise ValueError("The provided matrix is singular and cannot be inverted.")


def get_mean_std_dic_for_channels(channel_lists,resampled_RawObs_testing_site_input_data,
                                   resampled_RawObs_training_site_input_data,
                                   buffer_radius_list, desire_year_list,
                                   MONTH_lists, BLCO_kfold):
    
    from itertools import product
    import numpy as np

    # NaN-safe? set these to np.nansum and a custom nan-count if needed
    SUM = np.sum

    mean_values = {}
    std_values  = {}

    for ch in channel_lists:
        s = 0.0          # sum
        ss = 0.0         # sum of squares
        n = 0            # count

        rec_ch = resampled_RawObs_testing_site_input_data[ch]  # bind once

        # Iterate without building lists or concatenating
        for br in buffer_radius_list:
            rec_br = rec_ch[br]
            for yr in desire_year_list:
                rec_yr = rec_br[yr]
                for mo in MONTH_lists:
                    rec_mo = rec_yr[mo]
                    for ifold in range(BLCO_kfold):
                        x = np.asarray(rec_mo[ifold])          # ensure ndarray (view if possible)
                        # If you know x has no NaNs:
                        s  += x.sum(dtype=np.float64)
                        ss += (x * x).sum(dtype=np.float64)
                        n  += x.size

                        # If NaN-safe is required, replace the three lines above with:
                        # x = x.ravel()
                        # mask = np.isfinite(x)
                        # x = x[mask]
                        # s  += x.sum(dtype=np.float64)
                        # ss += (x * x).sum(dtype=np.float64)
                        # n  += x.size

        if n == 0:
            mean_values[ch] = np.nan
            std_values[ch]  = np.nan
        else:
            mu   = s / n
            var  = max(ss / n - mu * mu, 0.0)  # numerical guard
            mean_values[ch] = mu
            std_values[ch]  = np.sqrt(var)
    from itertools import product
    import numpy as np

    # NaN-safe? set these to np.nansum and a custom nan-count if needed
    SUM = np.sum

    mean_values = {}
    std_values  = {}

    for ch in channel_lists:
        s = 0.0          # sum
        ss = 0.0         # sum of squares
        n = 0            # count

        rec_ch = resampled_RawObs_testing_site_input_data[ch]  # bind once

        # Iterate without building lists or concatenating
        for br in buffer_radius_list:
            rec_br = rec_ch[br]
            for yr in desire_year_list:
                rec_yr = rec_br[yr]
                for mo in MONTH_lists:
                    rec_mo = rec_yr[mo]
                    for ifold in range(BLCO_kfold):
                        x = np.asarray(rec_mo[ifold])          # ensure ndarray (view if possible)
                        # If you know x has no NaNs:
                        s  += x.sum(dtype=np.float64)
                        ss += (x * x).sum(dtype=np.float64)
                        n  += x.size

        if n == 0:
            mean_values[ch] = np.nan
            std_values[ch]  = np.nan
        else:
            mu   = s / n
            var  = max(ss / n - mu * mu, 0.0)  # numerical guard
            mean_values[ch] = mu
            std_values[ch]  = np.sqrt(var)
            
            
    ## normalize the training data
    for ch in channel_lists:
        mu = mean_values[ch]
        sigma = std_values[ch]
        for br in buffer_radius_list:
            for yr in desire_year_list:
                for mo in MONTH_lists:
                    for ifold in range(BLCO_kfold):
                        resampled_RawObs_testing_site_input_data[ch][br][yr][mo][ifold] = (resampled_RawObs_testing_site_input_data[ch][br][yr][mo][ifold] - mu)/sigma
                        resampled_RawObs_training_site_input_data[ch][br][yr][mo][ifold] = (resampled_RawObs_training_site_input_data[ch][br][yr][mo][ifold] - mu)/sigma
    return resampled_RawObs_testing_site_input_data, resampled_RawObs_training_site_input_data