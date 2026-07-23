import numpy as np

def Bilinearinterpolate_GC_concentraion(Cxfyf:np.array,Cxfyc:np.array,Cxcyf:np.array,Cxcyc:np.array,delta_x:float,delta_y:float,dx:np.array,dy:np.array):
    Cx1 = Cxcyf * (dx/delta_x) + Cxfyf  * (1-dx/delta_x)
    Cx2 = Cxcyc * (dx/delta_x) + Cxfyc  * (1-dx/delta_x)
    Concentration = Cx2 * (dy/delta_y) + Cx1 * (1-dy/delta_y)
    return Concentration

def calculate_difference(GeoLAT,GeoLON,GCLAT,GCLON,ix,jy,lat_floor_indices,lon_floor_indices):
    dx = GeoLAT[ix] - GCLAT[lat_floor_indices[ix,jy]]
    dy = GeoLON[jy] - GCLON[lon_floor_indices[ix,jy]]
    return dx, dy

def get_BilinearInterpolate_Index(fine_Lat, fine_Lon, fine_Lat_map, fine_Lon_map, coarse_Lat, coarse_Lon):

    Coarse_Lat_Delta = coarse_Lat[2]-coarse_Lat[1]
    Coarse_Lon_Delta = coarse_Lon[2]-coarse_Lon[1]
    Coarse_Lat_min   = coarse_Lat[0]
    Coarse_Lon_min   = coarse_Lon[0]

    lat_floor_array = np.floor((fine_Lat - Coarse_Lat_min)/Coarse_Lat_Delta)
    lat_ceil_array  = np.ceil((fine_Lat - Coarse_Lat_min)/Coarse_Lat_Delta)
    lat_nearest_array = np.round((fine_Lat - Coarse_Lat_min) / Coarse_Lat_Delta)
    lon_floor_array = np.floor((fine_Lon - Coarse_Lon_min)/Coarse_Lon_Delta)
    lon_ceil_array  = np.ceil((fine_Lon - Coarse_Lon_min)/Coarse_Lon_Delta)
    lon_nearest_array = np.round((fine_Lon - Coarse_Lon_min) / Coarse_Lon_Delta)

    lat_floor_array = lat_floor_array.astype(int)
    lat_ceil_array = lat_ceil_array.astype(int)
    lat_nearest_array = lat_nearest_array.astype(int)
    lon_floor_array = lon_floor_array.astype(int)
    lon_ceil_array  = lon_ceil_array.astype(int)
    lon_nearest_array = lon_nearest_array.astype(int)

    dx = np.zeros((len(fine_Lat),len(fine_Lon)),dtype=np.float32)
    dy = np.zeros((len(fine_Lat),len(fine_Lon)),dtype=np.float32)
    for ix in range(len(fine_Lat)):
        print('ix: ',ix)
        dy[ix,:] = fine_Lon_map[ix,:]-coarse_Lon[lon_floor_array]
    for iy in range(len(fine_Lon)):
        print('iy: ',iy)
        dx[:,iy] = fine_Lat_map[:,iy]-coarse_Lat[lat_floor_array]
    return lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy
