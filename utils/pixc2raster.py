### author: xin luo, 
### create: 2025.7.28
### des: convert pixc data to raster format


import numpy as np
import xarray as xr
from sklearn.neighbors import KDTree

def pixc2raster(pixc_var, pixc_lonlat, raster_extent, resolution):
    """
    des: convert pixc data to raster format, using the median height of neighbors.
    params:
        pixc_var: pixc data variable (e.g., height) as a numpy array.
        pixc_lonlat: tuple/list of two numpy arrays (longitude, latitude) of the pixc data.
        raster_extent: tuple/list, raster data extent (min_lon, max_lon, min_lat, max_lat).
        resolution: int or list/tuple (lon_resolution, lat_resolution).
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
    var_neighbors_median_grid = []
    for i, neighbors in enumerate(pixc_var_neighbors):
        if np.isnan(neighbors).all():
            var_neighbors_median_grid.append(np.nan)
        else:
            var_neighbors_median_grid.append(np.nanmedian(neighbors))
    var_neighbors_median_grid = np.array(var_neighbors_median_grid).reshape(lats_grid.shape)

    ### Create the DataArray
    coords = {'x': xs_linspace, 'y': ys_linspace}
    raster_da = xr.DataArray(var_neighbors_median_grid, 
                                coords=coords, 
                                dims=['y', 'x'])

    return raster_da



