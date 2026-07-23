import numpy as np
import math
import toml
import os
from scipy.spatial.distance import cdist

# Filename tag for map-side uncertainty outputs. Env-overridable so a v7.2 run
# can tag its outputs (e.g. 'cf_v7_filtered') without disturbing the v5/v6 default.
Obs_version = os.environ.get('MAHAL_OBS_VERSION', 'cf_v6_filtered')

# Training data directories.
# /path/to is the path inside the Docker container (LSF jobs).
# On the host (e.g. running a notebook on the login node), fall back to the
# storage1 mount.
_CANDIDATE_ROOTS = [
    '/path/to/NO2_DL_global/Training_Evaluation_Estimation/',
    '/path/to/NO2_DL_global/Training_Evaluation_Estimation/',
    '/path/to/NO2_DL_global/Training_Evaluation_Estimation/',
]
def _resolve_root():
    for d in _CANDIDATE_ROOTS:
        if os.path.isdir(d):
            return d
    # If nothing resolves (e.g. brand-new year), default to container path so
    # writes under LSF still work.
    return _CANDIDATE_ROOTS[0]

Resampled_Training_BLISCO_data_outdir = _resolve_root()
Figure_outdir = Resampled_Training_BLISCO_data_outdir

def neighbors_haversine_indices(train_lat, train_lon, test_lat, test_lon, k):
    try:
        from sklearn.neighbors import NearestNeighbors
        train_rad = np.c_[np.radians(train_lat), np.radians(train_lon)]
        test_rad  = np.c_[np.radians(test_lat),  np.radians(test_lon)]
        k = int(min(k, len(train_rad)))
        if k <= 0:
            return np.empty((len(test_rad), 0), dtype=np.int64)
        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='haversine')
        nn.fit(train_rad)
        _, idx = nn.kneighbors(test_rad, n_neighbors=k, return_distance=True)
        return idx.astype(np.int64, copy=False)
    except Exception:
        # Fallback uses the pure NumPy implementation defined above
        return batch_topk_indices(test_lat, test_lon, train_lat, train_lon, k)

def Get_typeName(bias, normalize_bias, normalize_species, absolute_species, log_species, species):
    if bias == True:
        typeName = '{}-bias'.format(species)
    elif normalize_bias:
        typeName = 'Normalized-{}-bias'.format(species)
    elif normalize_species == True:
        typeName = 'Normaized-{}'.format(species)
    elif absolute_species == True:
        typeName = 'Absolute-{}'.format(species)
    elif log_species == True:
        typeName = 'Log-{}'.format(species)
    return  typeName