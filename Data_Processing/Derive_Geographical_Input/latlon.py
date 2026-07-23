import numpy as np
import os

spherical_coordinates_outdir = '/path/to/NO2_DL_global/input_variables/Geographical_Variables_input/Spherical_Coordinates/'

# Latitude and longitude in degrees
lat = np.linspace(-59.995, 69.995, 13000)
lon = np.linspace(-179.995, 179.995, 36000)

# Convert to radians
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Create 2D mesh
lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

# Convert to 3D Cartesian coordinates
x = np.cos(lat_grid) * np.cos(lon_grid)
y = np.cos(lat_grid) * np.sin(lon_grid)
z = np.sin(lat_grid)

if not os.path.isdir(spherical_coordinates_outdir):
    os.makedirs(spherical_coordinates_outdir)

S1_outfile = spherical_coordinates_outdir + 'Spherical_Coordinates_1.npy'
S2_outfile = spherical_coordinates_outdir + 'Spherical_Coordinates_2.npy'
S3_outfile = spherical_coordinates_outdir + 'Spherical_Coordinates_3.npy'
np.save(S1_outfile, x)
np.save(S2_outfile, y)
np.save(S3_outfile, z)