import numpy as np
import os
from itertools import product

def Get_Mahalanobis_Distances(MONTHS, buffer_radius_list, startyear, endyear,
                              BLISCO_obs_data_recording,BLISCO_final_data_recording,
                              EachMonth_EachYear_Martix_Mahalanobis_distance_recording,
                              EachMonth_AllYear_Martix_Mahalanobis_distance_recording,
                              AllMonth_AllYear_Martix_Mahalanobis_distance_recording):
    years = [str(y) for y in range(startyear, endyear + 1)]
    # ---- 1) BLISCO obs/final ----
    obs_parts = []
    final_parts = []
    for r, y, m in product(buffer_radius_list, years, MONTHS):
        obs_parts.append(BLISCO_obs_data_recording[r][y][m])
        final_parts.append(BLISCO_final_data_recording[r][y][m])

    # If each piece is a 1D array, concatenate. If scalars, hstack also works.
    try:
        All_BLISCO_obs_data   = np.concatenate(obs_parts, axis=0)
        All_BLISCO_final_data = np.concatenate(final_parts, axis=0)
    except Exception:
        # Fallback if shapes are irregular / scalars
        All_BLISCO_obs_data   = np.hstack(obs_parts).ravel()
        All_BLISCO_final_data = np.hstack(final_parts).ravel()

    # ---- 2) Mahalanobis distances ----
    eme_parts = []  # EachMonth_EachYear
    emay_parts = [] # EachMonth_AllYear
    aamay_parts = []# AllMonth_AllYear

    for r, y, m in product(buffer_radius_list, years, MONTHS):
        eme_parts.append(EachMonth_EachYear_Martix_Mahalanobis_distance_recording[r][y][m])
        emay_parts.append(EachMonth_AllYear_Martix_Mahalanobis_distance_recording[r][y][m])
        aamay_parts.append(AllMonth_AllYear_Martix_Mahalanobis_distance_recording[r][y][m])

    try:
        All_EachMonth_EachYear_Martix_Mahalanobis_distance = np.concatenate(eme_parts, axis=0)
        All_EachMonth_AllYear_Martix_Mahalanobis_distance  = np.concatenate(emay_parts, axis=0)
        All_AllMonth_AllYear_Martix_Mahalanobis_distance   = np.concatenate(aamay_parts, axis=0)

    except Exception:
        All_EachMonth_EachYear_Martix_Mahalanobis_distance = np.hstack(eme_parts).ravel()
        All_EachMonth_AllYear_Martix_Mahalanobis_distance  = np.hstack(emay_parts).ravel()
        All_AllMonth_AllYear_Martix_Mahalanobis_distance   = np.hstack(aamay_parts).ravel()

    return All_BLISCO_obs_data, All_BLISCO_final_data, All_EachMonth_EachYear_Martix_Mahalanobis_distance, All_EachMonth_AllYear_Martix_Mahalanobis_distance, All_AllMonth_AllYear_Martix_Mahalanobis_distance