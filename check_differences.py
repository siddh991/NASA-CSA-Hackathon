import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
def display_ndvi(ndvi_arr):
    im = plt.imshow(ndvi_arr)
    plt.colorbar(im)



def get_differences(file1, file2):
    dataset2 = rio.open('2019-08-02-0736.tif')
    dataset1 = rio.open('2019-08-01-0523.tif')
    band1 = dataset.read(1)
    band2 = dataset2.read(1)
    count1 = (band1 <0.2 ).sum()
    count2 = (band2 <0.2 ).sum()
    percent_increase = (count2 / count1)*100
    return percent_increase


