### author: xin luo, 
### create: 2025.7.28
### des: convert pixc data to raster format


import numpy as np
import xarray as xr
from sklearn.neighbors import KDTree

def pixc2raster(pixc_var, pixc_lonlat, raster_extent, resolution, agg_method = 'median'):
    """
    des: convert pixc data to raster format, using the median height of neighbors.
    params:
        pixc_var: pixc data variable (e.g., height) as a numpy array.
        pixc_lonlat: tuple/list of two numpy arrays (longitude, latitude) of the pixc data.
        raster_extent: tuple/list, raster data extent (min_lon, max_lon, min_lat, max_lat).
        resolution: int or list/tuple (lon_resolution, lat_resolution).
        agg_method: str, aggregation method ('mean' or 'median').
    returns: xarray DataArray in raster format.
    """
    if isinstance(resolution, int) or isinstance(resolution, float):
        res_lon = res_lat = resolution
    else: res_lon, res_lat = resolution
    lons_pixc, lats_pixc = pixc_lonlat
    ## Calculate the grid size
    xmin, xmax, ymin, ymax = raster_extent
    width = int((xmax - xmin) / res_lon) + 1
    height = int((ymax - ymin) / res_lat) + 1    
    xs_linspace = np.linspace(xmin, xmax, width)
    ys_linspace = np.linspace(ymax, ymin, height)    
    lons_grid, lats_grid = np.meshgrid(xs_linspace, ys_linspace)
    lon_lat_grid = np.column_stack((lons_grid.ravel(), lats_grid.ravel()))
    tree = KDTree(data=list(zip(lons_pixc, lats_pixc)))
    ids, distance = tree.query_radius(X=lon_lat_grid, 
                                        r = np.max([res_lat, res_lon]), 
                                        return_distance=True)
    ## Calculate the median value of neighbors
    pixc_var_neighbors = [pixc_var[id_list] for id_list in ids]
    pixc_var_neighbors = [np.nan if neighbors.size==0 else neighbors 
                                            for neighbors in pixc_var_neighbors]
    if agg_method == 'median':
        var_neighbors_agg_grid = [np.nanmedian(neighbors) if not np.isnan(neighbors).all() else np.nan
                                   for neighbors in pixc_var_neighbors]
    elif agg_method == 'mean':
        var_neighbors_agg_grid = [np.nanmean(neighbors) if not np.isnan(neighbors).all() else np.nan
                                   for neighbors in pixc_var_neighbors]
    else:
        raise ValueError("agg_method should be 'mean' or 'median'")
    var_neighbors_agg_grid = np.array(var_neighbors_agg_grid).reshape(lats_grid.shape)

    ### Create the DataArray
    coords = {'x': xs_linspace, 'y': ys_linspace}
    raster_da = xr.DataArray(var_neighbors_agg_grid, 
                                coords=coords, 
                                dims=['y', 'x'])

    return raster_da



