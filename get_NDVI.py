import gdal, numpy, math, os, random
import matplotlib.pyplot as plt
import pandas as pd
import wget

def fix_nodata(raster):
    conn = gdal.Open(raster)
    fix = gdal.Translate(raster.replace(".TIF", "_nodata_fixed.TIF"), conn,noData=0)
    return raster.replace(".TIF", "_nodata_fixed.TIF")

def RasterToNumPyArray(raster_path, dtype=None):
    if dtype == None:
        return gdal.Open(raster_path).ReadAsArray()
    else:
        return gdal.Open(raster_path).ReadAsArray().astype(dtype)
    
def dn_to_radiance(band, band_number):
    gain = {3:0.621654, 4:0.639764}
    bias = {3:-5.62, 4:-5.74}
    radiance = band * gain[band_number]  + bias[band_number]
    return radiance 

def get_sun_angle(metadata_file):
    f = open(metadata_file, 'r')
    metadata = f.read().split("\n")
    for line in metadata:
        if "SUN_ELEVATION" in line:
            f.close()
            return float(line.split("=")[1].strip())
        
def get_earth_sun_distance(metadata_file):
    f = open(metadata_file, 'r')
    metadata = f.read().split("\n")
    for line in metadata:
        if "EARTH_SUN_DISTANCE" in line:
            f.close()
            return float(line.split("=")[1].strip())
        
def radiance_to_reflectance(band, band_number, sun_angle, earth_sun_distance):
    sun_radiance = {3:0.621654, 4:0.639764}
    numerator = math.pi * band * (earth_sun_distance ** 2)
    denominator = sun_radiance[band_number] * math.sin(sun_angle * math.pi / 180)
    reflectance = numerator / denominator
    return reflectance

def enforce_positive_reflectance(band):
    band[numpy.where(band < 0)] = 0 
    band[numpy.where(band == 0)] = random.random() * 0.001
    return band

def calculate_ndvi(band3, band4):
    ndvi = (band4 - band3) / (band4 + band3)
    return ndvi

def display_ndvi(ndvi_arr):
    im = plt.imshow(ndvi_arr)
    plt.colorbar(im)
    
def construct_ndvi(band3_raw, band4_raw, mtl):
    dataset = gdal.Open(band3_raw, gdal.GA_ReadOnly)
    if dataset:
        print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
        print("Projection is {}".format(dataset.GetProjection()))
    band3_nozero = fix_nodata(band3_raw)
    band4_nozero = fix_nodata(band4_raw)
    band3_nparr = RasterToNumPyArray(band3_nozero)
    band4_nparr = RasterToNumPyArray(band4_nozero)
    band3_radiance = dn_to_radiance(band3_nparr, 3)
    band4_radiance = dn_to_radiance(band4_nparr, 4)
    sun_angle = get_sun_angle(mtl)
    earth_sun_distance = get_earth_sun_distance(mtl)
    band3_reflectance = radiance_to_reflectance(band3_radiance, 3, sun_angle, earth_sun_distance)
    band4_reflectance = radiance_to_reflectance(band4_radiance, 4, sun_angle, earth_sun_distance)
    band3_allpos = enforce_positive_reflectance(band3_reflectance)
    band4_allpos = enforce_positive_reflectance(band4_reflectance)
    ndvi = calculate_ndvi(band3_allpos, band4_allpos)
    return ndvi
    