import shutil
from pathlib import Path
from osgeo import gdal, osr
from fastcore.utils import *

import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, Polygon


def close_holes(polygon):
    """
    Close polygon holes by limitation to the exterior ring.
    Args:
        polygon: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    """
    if polygon.interiors:
        return Polygon(list(polygon.exterior.coords))
    else:
        return polygon


def pixel_mask_to_geodataframe(pixel_mask, reference_image_filepath, image_id, target_crs, bg_threshold=0.5, min_area=40):
    """
    Converting pixel mask to vector polygons. Works similar
    to sol.vector.mask.mask_to_poly_geojson but with better handling for min_area
    :param pixel_mask: np array of predicted segmentation mask
    :param reference_image_filepath: georeferenced satellite image that mask is predicted on
    :param image_id: id of tile e.g. AOI_3_Vegas_xxx
    :param target_crs: reference system that has units of meters and is suitable for the reference image
    :param bg_threshold: threshold value in which to consider the pixel to have the positive class
    :param min_area: min area of polygon to consider (filter out the really tiny polygons that are probably noise)
    :return: geodataframe with the predicted polygons in WKT
    """
    with rasterio.open(str(reference_image_filepath)) as ref:
        transform = ref.transform
        crs = ref.crs
        tile_bounds = ref.bounds
        ref.close()

    mask = pixel_mask/pixel_mask.max() > bg_threshold
    shapes = features.shapes(pixel_mask, mask=mask, transform=transform)
    shapes_px = features.shapes(pixel_mask, mask=mask)


    geometries = [shape(i[0]) for i in list(shapes)]
    geometries_pixel = [shape(i[0]) for i in list(shapes_px)]
    output_gdf = gpd.GeoDataFrame({'geometry': geometries})
    output_gdf['PolygonWKT_Pix'] = geometries_pixel
    output_gdf.crs = crs

    output_gdf_proj = output_gdf.to_crs(epsg=target_crs)
    output_gdf_proj = output_gdf_proj[output_gdf_proj.area > min_area]
    output_gdf_proj.geometry = output_gdf_proj.geometry.apply(lambda p: close_holes(p))
    output_gdf_proj["image_filepath"] = reference_image_filepath
    output_gdf_proj["ImageId"] = image_id
    output_gdf_proj["Confidence"] = 1
    return output_gdf_proj.to_crs(crs)


def georeference_image(image_input_filepath, image_output_filepath, image_width, image_height, map_bottom, map_top, map_left, map_right):
    dataset = gdal.Open(str(image_input_filepath), gdal.GA_Update)
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    # Enter the GCPs
    #   Format: [map x-coordinate(longitude)], [map y-coordinate (latitude)], [elevation],
    #   [image column index(x)], [image row index (y)]
    gcps = [
        gdal.GCP(map_left, map_bottom, 0, 0, image_height),
        gdal.GCP(map_right, map_bottom, 0, image_width, image_height),
        gdal.GCP(map_right, map_top, 0, image_width, 0),
        gdal.GCP(map_left, map_top, 0, 0, 0)]

    # Apply the GCPs to the open output file:
    dataset.SetGCPs(gcps, sr.ExportToWkt())
    gdal.Warp(str(image_output_filepath), dataset, dstSRS='EPSG:4326', format='gtiff')

    # Close the output file in order to be able to work with it in other programs:
    dataset = None


