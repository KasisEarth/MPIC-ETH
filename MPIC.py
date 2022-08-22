# -*- coding: utf-8 -*-
"""
Multiple Pairwise Image Correlation Routine

Based on the FFT DIC routine:
    Bickel, VT, S. Loew, J. Aaron, and N. Goedhart. 2022.
    “A Global Catalog of Lunar Granular Flow Features.” 
    LPI Contributions 2678:1824.
    
and based on LAMMA routine:
    Dematteis, Niccolò, Daniele Giordan, Bruno Crippa,
    and Oriol Monserrat. 2022. “Fast Local Adaptive Multiscale Image Matching
    Algorithm for Remote Sensing Image Correlation.”
    Computers & Geosciences 159:104988. doi: 10.1016/j.cageo.2021.104988.

Author: Lukas Frei
        Master Project D-ERDW ETHZ 2022
"""

# Import modules which are needed in the main file

import time         # Needed to get the runtime
import cv2          # module for image reading and image manipulation
from pathlib import Path    

# Import of classes for MPIC routin

from preprocessing import PreProcessing
from dic import PixOff
from georectification import Georect
from postprocessing import PostProcessing
from timeseries import TimeSeries


##############################################################################
''' USERS DEFINE VARIABLES HERE '''
##############################################################################
# parameters of correlation methods and filters can be set in the classes
# see the files "preprocessing", "dic", "postprocessing"

# Inputs Wallis filter and image reprentation
WALLIS = False          # True = wallis filtering on

# Image representation (Dematteis & Gordian 2021)
REPRESENTATION = 'or'   # = 'or' -> transformation from 'in' to 'or' (only for fft)
                        # = 'in' -> no transformatiopn

# similarity function algorithm (zncc and fft use lamma approach, Dematteis & Gordian 2022)
METHOD = 'fft'          # = 'fft' -> fast fourier transform
                        # = 'zncc' -> zero-mean normalized cross-correlation
                        # = 'cxc' -> cosine similarity

# baseline of the pairwise image matching, minimum and maximum
BAS_MIN = 1
BAS_MAX = 3
               
# Co-registration settings
CROPPING = 0    # 0 = full image for co-registration
                # 1 = define patch for coregistration    
                # 2 = a predefined patch with indeces defined below will be used
                
# indeces of path for co-registration, only if CROPPING = 2, [upper,lower,left,right]
CROPIND = [300,1250,1700,2500]    


# Filter settings
filtertype = 5    # 1 = threshold filter: error
                  # 2 = threshold filter: magnitude
                  # 3 = arithmetic mean filter
                  # 4 = vector filter
                  # 5 = time series filter 1 (error and magnitude combined)
         
plottype = 2      # 0 = no plot
                  # 1 = scatter plot
                  # 2 = quiver plot
                  
# if True, all filtered results in the output folder get georectified.
georect = True

# if True, 3D results are combined into a time series
timeser = True

# if true, time series filter will be applied
ts_filt = True 

# if True, a mask saved in the data folder will be used to mask the filtered results
masking = True
            

##############################################################################
''' READ FILE NAMES IN INPUT FOLDER '''
##############################################################################

# path to the input folder
in_path = Path(__file__).parent.joinpath("input")
# path to the output folder
out_path = in_path.parent.joinpath("output")
# path to the image output folder
img_path = out_path.joinpath("images")
# path to the georect_data folder
data_path = Path(__file__).parent.joinpath("additional_data")

input_list = [p for p in in_path.iterdir() if p.suffix.lower() == '.jpg']

# input_list = [input_list[0], input_list[-5]]
# input_list = input_list[6:]

##############################################################################
''' MULTIPLE PAIRIWISE IMAGE MATCHING '''
##############################################################################

counter = 0
for i in range(len(input_list)):
    for j in range(i+BAS_MIN,i+BAS_MAX):

        p_name = input_list[i]
        try:
            s_name = input_list[j]
        except:
            continue
        out_fn = str(p_name.stem) + '-' + str(s_name.stem) + '.csv'
        
        # insert wallis into output filename if it is applied
        if WALLIS == True:
            out_fn = 'wallis-' + out_fn
        
        print('\n<<< Running # ', counter, ' >>>')
        counter += 1      
        
        primary = cv2.imread(str(p_name), cv2.IMREAD_GRAYSCALE)
        secondary = cv2.imread(str(s_name), cv2.IMREAD_GRAYSCALE)
        
        ##############################################################################
        ''' PRE-PROCESSING '''
        ##############################################################################
        
        print('<<< Pre-processing >>>')
        start_time = time.time()
        
        # approaches using lamma will automatically apply "or" representation if possible
        if METHOD == 'zncc' or METHOD == 'cxc':
            REPRESENTATION = 'in'
            
        # Initiate class PreProcessing, automatically does wallis filtering and OR 
        # representation if assigned so in the beginning
        prepro = PreProcessing(primary,secondary,WALLIS,REPRESENTATION)
        
        # return preprocessed primary or secondary image with getter method.
        t0,t1 = prepro.primary(),prepro.secondary()
        del prepro
            
        print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        ##############################################################################
        ''' CO-REGISTRATION '''
        ##############################################################################
        
        print('<<< Co-registration >>>')
        
        # Initiate calss PixOff, this only needs primary and secondary as input
        trackPixels = PixOff(t0,t1)
        
        # Start co-regristration with inputs: cropping, path to inputs, oversampling, splitting factor
        trackPixels.coregistration(CROPPING,str(p_name),CROPIND)
        
            
        print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        ##############################################################################
        ''' DIGITAL IMAGE CORRELATION '''
        ##############################################################################
        
        print('<<< Running DIC >>>')
        
        trackPixels.pixelOffset(METHOD)
        
        results = trackPixels.results
        WINDOWSIZE = trackPixels.WIN
        
        # get the template size and step size of the method used for output name
        if METHOD == 'fft':
            tmpsz = str(trackPixels.WIN)
            stepsize = trackPixels.FFTSTEP # needed for filtering
            exname = str(out_path.joinpath(METHOD + '-' + tmpsz + '-' + 'step-' + str(stepsize) + '-' + out_fn))
            imgname = str(img_path.joinpath(METHOD + '-' + tmpsz + '-' + 'step-' + str(stepsize) + '-' + out_fn))
            filt_exname = str(out_path.joinpath('filtered-' + METHOD + '-' + tmpsz + '-' + 'step-' + str(stepsize) + '-' + out_fn))
        elif METHOD == 'zncc' or METHOD == 'cxc':
            tmpsz = str(trackPixels.NODEDIST)
            info_sz = str(trackPixels.MAXSCALE) + '-' + str(trackPixels.NODEDIST) +'-tilesize-' + str(trackPixels.TILESIZE)
            stepsize = trackPixels.NODEDIST # needed for filtering
            exname = str(out_path.joinpath(METHOD + '-' + info_sz + '-' + out_fn))
            imgname = str(img_path.joinpath(METHOD + '-' + info_sz + '-' + out_fn))
            filt_exname = str(out_path.joinpath('filtered-' + METHOD + '-' + info_sz + '-' + out_fn))
            
        # save the resulting pandas DataFrame to csv
        results.to_csv(exname,index=False)
        
        print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        ##############################################################################
        ''' POST-PROCESSING '''
        ##############################################################################
        
        print('<<< Post-Processing >>>')
        
        # call co-registered secondary for plotting
        # secondary = trackPixels.secondary
        
        
        postpro = PostProcessing(results,primary,secondary,tmpsz,stepsize,filtertype,plottype)
        
            
        # start filtering
        postpro.filter(masking,data_path)
        
        # if filter is apllied, get results and save them 
        if filtertype != 0:
            # get filtered results with getter function 
            filt_res = postpro.results
            
            # save filtered pandas DataFrame
            filt_res.to_csv(filt_exname,index=False)
        
        
        # create image output folder if it doesn't exist
        img_path.mkdir(parents=True, exist_ok=True)
        
        # plot the results based on the plottype variable
        postpro.plotter(imgname)
        
        # create background images
        postpro.createBG(s_name, img_path)
        
        print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

    
##############################################################################
''' CREATE TIME SERIES '''
##############################################################################
if timeser:
    print('\n<<< Create Time Series >>>')
    # Creates per day the average velocity from all results that include this 
    # day (date in between the primary and secondary image)
    
    # initiate timeseries class, input is the path to the georectified 3D results
    createSeries = TimeSeries(out_path,ts_filt)
    
    # create a time series based on hourly time windows (px/h), 
    # then calculate the mean hourly verlocity for every day
    # then calculates the cumulated displacement on daily displacement
    createSeries.create_ts()

    print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
##############################################################################
''' GEO-RECTIFICATION '''
##############################################################################
if georect:
    print('\n<<< Georectification >>>')
    
    # initilize the Georect class with the paths to the georect matrices and DIC results
    rectification = Georect(data_path,Path(out_path).joinpath('time-series'))
    
    # rectify all files in the output folder
    rectification.rectify()
    
    print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
        
##############################################################################
''' EXTRACT POINT INFORMATION '''
##############################################################################
if timeser and georect:
    # extract the 3D cumulative information at the locations given as inputs
    # e.g. extractTS('name of point', 1000,1000)
    
    print('\n<<< Extract points from cumulative displacements >>>')
    
    # example point
    createSeries.extract_pointTS('point 01-128-128',128,128)


    print('\n<<< Create a plot from every extracted point >>>')

    createSeries.createPointPlot()


    
##############################################################################
''' CREATE A GIF OF THE TIME SERIES (displacement and inclination) '''
##############################################################################
if timeser and georect:
    print('\n<<< CREATE GIF >>>')
    
    # creates for each day of the time series an image and merges them in a gif
    createSeries.createGIF(data_path)
    createSeries.extract_decorr()
    createSeries.inclinationGIF()
    
    # example point
    createSeries.extract_incli('point 01-128-128',128,128)

    
    print("--- RUNTIME %s seconds ---" % (time.time() - start_time))
    start_time = time.time()