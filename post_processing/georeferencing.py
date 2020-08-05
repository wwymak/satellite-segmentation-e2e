import shutil
from osgeo import gdal, osr
from fastcore.utils import *


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


