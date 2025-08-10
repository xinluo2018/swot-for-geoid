### author: xin luo 
### create: 2025.2.9, modified: 2025.8.5
### des: read the SWOT data and select the variables within the given geometry.

import geopandas as gpd
from pyproj import CRS
import numpy as np

def swot_pixc_mask(pixc_nc, 
                    vars_sel=['latitude', 'longitude', 'height', 
                              'solid_earth_tide', 'pole_tide', 
                              'load_tide_fes', 'geoid',
                              ], 
                    region_gdf=None, 
                    path_masked=None):
    """
    des: read the SWOT_L2_HR_PIXC data and select the variables within the given geometry.    
    args:
      pixc_nc (xarray): the SWOT PIXC dataset.
      vars (list): list of variables to be selected.
      region_gpd (geopandas): the geometry to be used for selecting the data.    
    returns:
      pix_masked (xarray): the selected variables within the geometry.
    """
        
    # 1. extract the variables from the dataset
    if vars_sel is not None:
        pixc_vars = pixc_nc[vars_sel].copy(deep=False)
    else:
        pixc_vars = pixc_nc.copy(deep=False)
    # 2. extract the points within the region
    if region_gdf is not None:
        if region_gdf.crs.to_epsg() != 4326:
            region_gdf = region_gdf.to_crs(epsg='4326')
        
        ## 2.1 find points within the bounding box of the region
        minx, miny, maxx, maxy = region_gdf.total_bounds
        lon = pixc_vars.longitude.values
        lat = pixc_vars.latitude.values

        ### find points within the bounding box
        bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        
        if not np.any(bbox_mask):
            print("No points within the region bounding box")
            return None
            
        pixc_subset = pixc_vars.isel(points=bbox_mask)

        # 2.2 find points within the precise geometry
        points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                pixc_subset.longitude.values, 
                pixc_subset.latitude.values
            )
        )
        
        id_points_in_geom = points.sindex.query(
            region_gdf.geometry.union_all(), 
            predicate='contains'
        )
        
        if len(id_points_in_geom) == 0:
            print("No points found within the precise geometry")
            return None
        pixc_masked = pixc_subset.isel(points=id_points_in_geom)

    # 3. check if any points remain after masking
    if pixc_masked.sizes['points'] == 0:
        print("No points found within the geometry")
        return None
    
    # 4. add acquisition date of the data
    pixc_masked.attrs = {"date": str(pixc_nc.illumination_time.values[0])[:10]}

    # 5. Write the selected variables to a new .nc file
    if path_masked is not None:
        if pixc_masked.sizes['points'] > 100000:
            encoding = {var: {'zlib': True, 'complevel': 4} for var in pixc_masked.data_vars}
            pixc_masked.to_netcdf(path_masked, encoding=encoding)
        else:
            pixc_masked.to_netcdf(path_masked)
        print(f"file written to {path_masked}")

    return pixc_masked



def swot_raster_mask(raster_nc, 
                     vars_sel=['x', 'y', 'wse','sig0', 'geoid', 'wse_qual'], 
                     region_gdf=None, 
                     path_masked=None):
    """
    des: read the SWOT_L2_HR_Raster data and select the variables within the given geometry.    
    args:
      raster_nc (xarray): the SWOT Raster dataset.
      vars (list): list of variables to be selected.
      region_gpd (geopandas): the geometry to be used for selecting the data.    
    returns:
      raster_masked (xarray): the selected variables within the geometry.
    """
    raster_masked = raster_nc.copy(deep=True)
    # Select main variables to be written out
    if vars_sel is not None:
        raster_masked = raster_masked[vars_sel]

    if region_gdf is not None:
        epsg_gdf = str(region_gdf.crs.to_epsg())
        crs_wkt = raster_nc.crs.attrs['crs_wkt']
        epsg_raster = str(CRS.from_wkt(crs_wkt).to_epsg())
        if epsg_gdf != epsg_raster:
          raise ValueError(f"EPSG code mismatch: {epsg_gdf} != {epsg_raster}. Please check the CRS of the data.")
        else:
          raster_masked = raster_masked.rio.clip(region_gdf.geometry.values, 
                                                 region_gdf.crs, 
                                                 drop=True,
                                                 all_touched=True)

    # Write the selected variables to a new .nc file
    if path_masked is not None:
        for var in raster_masked.data_vars:
            if 'grid_mapping' in raster_masked[var].attrs:
                del raster_masked[var].attrs['grid_mapping']
        raster_masked.to_netcdf(path_masked)
        print(f"file read and written to {path_masked}")
    return raster_masked
