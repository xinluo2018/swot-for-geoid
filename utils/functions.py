### author: xin luo, 
### create: 2024.6.9, modified: 2025.8.3
### des: some useful functions for swot data processing.

import numpy as np
import rioxarray   # rioxarray is used for reprojection and clipping of raster data
import math
from sklearn.neighbors import KDTree

def pixc_to_slant_raster(pixc, key='height'):
    '''
     des: convert the swot pixel data to slant range format.
    '''
    az = pixc.azimuth_index.astype(int)
    rng = pixc.range_index.astype(int)
    out_arr = np.zeros((pixc.interferogram_size_azimuth + 1, \
                        pixc.interferogram_size_range + 1)) + np.nan
    # handle complex interferogram
    if key=='interferogram':
        out_arr = out_arr.astype('complex64')
        var = pixc[key][:,0] + 1j * pixc[key][:,1]
    else:
        var = pixc[key]
    out_arr[az, rng] = var
    return out_arr

def pixc_geophy_cor(pixc_nc, height_key='height'):
    """
    des: geophysical corrections for the height data.
        According to the swot l2 hr pixc document(L2_HR_raster document and SWOT product handbook), 
        the dry troposphere, wet troposphere, and the ionosphere have been corrected for the height data,
        and only solid earth tide, pole tide, and load tide are not corrected for the height.
    """
    height = pixc_nc[height_key].values
    solid_tide = pixc_nc.solid_earth_tide.values
    pole_tide = pixc_nc.pole_tide.values
    load_tide = pixc_nc.load_tide_fes.values
    return height - (solid_tide + pole_tide + load_tide)    

def swot_raster_reproj(raster_nc, epsg_from, epsg_to):
    '''
    des: reproject the raster dataset from one EPSG code to another.
    args:
      raster_nc (xarray): the SWOT Raster dataset.
      epsg_from (str): the EPSG code of the original dataset.
      epsg_to (str): the EPSG code to which the dataset will be reprojected.
    returns:
      raster_reproj_nc (xarray): the reprojected raster dataset.
    '''
    ## Ensure the dataset is compatible with rioxarray
    if not raster_nc.rio.crs:
        raster_nc = raster_nc.rio.write_crs("EPSG:"+epsg_from)
    ## Reprojection
    raster_reproj_nc = raster_nc.rio.reproject("EPSG:"+epsg_to)
    return raster_reproj_nc

def deg2meter(degree, lat=0):
    """
    des: convert degree resolution to meter resolution
    params:
        degree (float): resolution in degrees
        lat (float): latitude of the location (default is equator)
    return:
        tuple: (distance in meters along longitude, and for latitude)
    """
    R = 6371000  # mean radius of the Earth in meters
    lat_distance_m = degree * (math.pi * R / 180)  # convert latitude resolution (constant)
    lon_distance_m = degree * (math.pi * R / 180) * math.cos(math.radians(lat))  # convert longitude resolution (varies with latitude)
    return (lon_distance_m, lat_distance_m)

def meter2deg(meter, lat=0):
    """
    des: convert meter resolution to degree resolution
    params:
        meter (float): distance in meters
        lat (float): latitude of the location (default is equator)    
    return:
        tuple: (distance in degrees for longitude, and for latitude)
    """
    # radius of the Earth in meters
    R = 6371000
    lat_distance_deg = (meter * 180) / (math.pi * R)  # convert latitude resolution (constant)
    lon_distance_deg = (meter * 180) / (math.pi * R * math.cos(math.radians(lat)))  # convert longitude resolution (varies with latitude)
    return (lon_distance_deg, lat_distance_deg)

def raster_directional_stats(raster_data, num_bins=36):
    """
    des: analyze the statistical characteristics of raster data in different directions.
    params 
        raster_data: Input raster data (2D array)
        num_bins: Number of angular bins (default 36, each 10 degrees)
    return: 
        angle array, mean array, std array
    """
    # 1. Create grid coordinates
    ny, nx = raster_data.shape
    x = np.arange(nx) - nx/2
    y = np.arange(ny) - ny/2
    xx, yy = np.meshgrid(x, y)
    
    # 2. Calculate polar coordinates for each pixel
    r = np.sqrt(xx**2 + yy**2)        # radius
    theta = np.arctan2(yy, xx)        # angle (radian), range [-π, π]
    theta = np.mod(theta, 2*np.pi)    # convert to [0, 2π]
    
    # 3. Create angle bins
    angle_bins = np.linspace(0, 2*np.pi, num_bins+1)        # angle bin edges
    bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2    # center of each bin
    
    # 4. Calculate statistics for each angle bin
    means = np.zeros(num_bins)
    stds = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Select pixels in the current angle bin
        mask = (theta >= angle_bins[i]) & (theta < angle_bins[i+1])
        # Exclude center point (radius=0)
        valid_mask = mask & (r > 0)
        
        # Calculate statistics
        if np.any(valid_mask):
            values = raster_data[valid_mask]
            means[i] = np.nanmean(values)
            stds[i] = np.nanstd(values)
        else:
            means[i] = np.nan
            stds[i] = np.nan
    
    # 5. Convert to degrees (0-360)
    bin_centers_deg = np.degrees(bin_centers)
    
    return bin_centers_deg, means, stds

def sample_from_raster(raster_value, raster_x, raster_y, points_x, points_y):
    x, y = np.meshgrid(raster_x, raster_y)
    raster_xy = np.column_stack([x.ravel(), y.ravel()])
    points_xy = [points_x, points_y]
    points_xy = np.column_stack(points_xy)
    ### build KDTree and query neighbors
    tree = KDTree(raster_xy)
    indices = tree.query(points_xy, k=1, return_distance=False).ravel()
    points_values = raster_value.ravel()[indices]
    return points_values

def hz01_hz20(data_01hz, time_01hz, time_20hz):
    '''
    des: convert 01hz data to 20hz data through time nearest interpolation. 
    '''
    time_20hz_ = np.expand_dims(time_20hz, axis=1)
    dif_time = abs(time_20hz_ - time_01hz)
    ind_min = dif_time.argmin(axis=1)
    data_20hz = data_01hz[ind_min]
    return data_20hz


