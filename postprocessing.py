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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import cv2
from scipy import ndimage
import numpy as np
import pandas as pd
import math
from pathlib import Path

class PostProcessing(object):
    '''
    A class with methods for filtering DIC results and plotting them
    
    ...
    
    Attributes
    ----------
    THR : float
        error masking threshold, values between 0-1 ( 1=no mask )
        
    MAG : float / int
        magnitude masking threshold, 2D displacement magnitudes larger than 
        MAG are filtered out
        when MAG = None --> MAG = half the window size by default.
        
    MFWS : float / int
        mean filter window size for arithmettic mean filter
    
    CUT : float / int, OPTIONAL 
        the window size divieded by CUT gives the value that will be used to cut
        DX and DY values that are larger than WIN/CUT
        CUT = 0 means no cut off
        
    PLTMIN : float / int
        minimum 2D displacement magnidute that is visualized in the plotting
        
    PLTMAX : float / int
        maximum 2D displacement magnitude that is visualized in the left plot
        only for visualization, when not defined (set PLTMAX = None), 
        it will be the 10th largest displacement measured
        
    VEC1 : float / int
        the window size diveded by VEC1 is the threshold which is used to filter
        magnitudes in the vector filter
        
    VEC2 : float / int
        the window size divided by VEC2 is the threshold which is used to filter
        DX and DY in the vector filter
        
    VEC3 : float / int
        threshold in the vector filter that is used to filter direciton changes
        in a 3x3 neighbourhood (in rad)
        
    MAXDECORR : float / int
        threshold, more decorrelated results in the masked results indicate
        that the pairing of the images is bad and the results are neglected
        
    MASKNAME : string
        name of the mask file in the additional_data folder, e.g. 'mask.csv'
    
    Methods
    -------
    filter : main method that calls the chosen filter by variable filtertype
    TS_filt1 : time series filter calibrated for Moosfluh case study
    thr_error_filt : error threshold filter
    thr_magn_filt : magnitude threshold filter
    mean_filt : mean filter, averaging displacement field
    vector_filt : vector filter, comparing averaged fields to local displacement value
    create_BG : create a background image for time series visualization
    plotter : plotting the filtered DIC results with points or arrows
    
    '''
    
    # error threshold filter settings
    THR = 1
    # magnitude threshold filter settings
    MAG = 32
    # mean filter settings
    MFWS = 6
    # vector filter settings
    CUT = 0
    VEC1 = 1
    VEC2 = 7.5
    VEC3 = 3
    # VEC1 = 9
    # VEC2 = 9.5
    # VEC3 = 3
    # plotting settings
    PLTMIN = 1
    PLTMAX = 15
    # maximum decorrelations allowed
    MASKNAME = 'mask.csv'
    MAXDECORR = 1000
    
    @property
    def results(self):
        """
        returns the filtered results matrix for further processing in the main script

        Returns
        -------
        np.array
            filtered results of DIC.

        """
        
        return self._filtres
    
    def __init__(self,results,primary,secondary,windowsize=64,stepsize=32,filtertype=1,plottype=1):
        """
        initiation method of the PostProcessing class

        Parameters
        ----------
        results : np.array
            results array of the DIC, columns: x, y, dx, dy, dtot, error.
            
        primary : np.array
            primary image as a np.array.
            
        secondary : np.array
            secondary image as a np.array.
            
        windowsize : int, optional
            window size used in the DIC. The default is 64.
            
        stepsize : int, optional
            Step size (nyquist, grid size) of the DIC. The default is 32.
            
        filtertype : int, optional
            values 1-5, defines the filter that will be used. The default is 1.
            1: error threshold filter
            2: magnitudes threshold filter
            3: mean filter
            4: vector filter
            5: time series filter
            
        plottype : int, optional
            values 1 or 2 define the type of plot visualized. The default is 1.
            1: points colored according the magnitude
            2: arrows with lengtth and color according the magnitude

        Returns
        -------
        None.

        """
        
        # primary and secondary image of the DIC
        self._secondary = secondary
        self._primary = primary
        
        # number defines the type of postprocessing filter
        self._filtertype = filtertype
        
        # save results of DIC as a class variable
        self._res = results
        
        # variable which will contain the filtered results later
        self._filtres = None
        
        # variable which will contain the filtered results for plotting later
        self._plotres = None
        
        # number defines type of plot
        self._plottype = plottype
        
        # defining window size of the DIC
        self.WIN = int(windowsize)
        
        if self.MAG == None:
            self.MAG = self.WIN/2
            
        # defining step size of the DIC
        self._stepsize = stepsize
        
        # saves the index of results that were filtered out
        self._filtout = np.array([0])
        # background image that will be saved in a defined folder with createBG
        self._BGimg = None
        
    def filter(self,masking=False,mask_path=None):
        """
        main filter method, calls the chosen filter method and saves the
        filtered results to a DataFrame variable

        Returns
        -------
        None.

        """
        
        if self._filtertype == 0:
            print('<<< No filter is selected (filtertype = 0) >>>')
            rr = np.array(self._res)
            
        if self._filtertype == 1:
            # thershold filter: error
            rr = self.thr_error_filt()
            
        if self._filtertype == 2:
            # thershold filter: magnitude
            rr = self.thr_mag_filt()
            
        if self._filtertype == 3:
            rr = self.mean_filt()
            
        if self._filtertype == 4:
            rr = self.vector_filt()
            
        if self._filtertype == 5:
            rr = self.TS_filt1()
        
        if masking:
            # read mask from the given path to the csv file
            mask = np.genfromtxt(str(mask_path.joinpath(self.MASKNAME)), delimiter=',')[1:,:] 
            
            if mask.shape == (1943,2592):
                bool_disp = np.ones((rr.shape[0],),dtype=bool)
                for ii in range( rr.shape[0]):
                    try:
                        if mask[int(rr[ii,1]),int(rr[ii,0])] == 1: 
                            bool_disp[ii] = False
                    except:
                        do_nothing=0
                mask = bool_disp
                
            rr[mask,2:] = np.nan
            
            print('# of decorrelated patches: ', rr[rr[:,-1]>1,-1].shape[0])
            
            if rr[rr[:,-1]>1,-1].shape[0]>self.MAXDECORR:
                # if there are too many decorrelations in the masked results, 
                # all results are neglected in the time series
                rr[:,2:] = np.nan
        
        # get variables from rr array
        x = rr[:,0]
        y = rr[:,1]
        dx = rr[:,2] 
        dy = rr[:,3]
        d2 = rr[:,4]
        error = rr[:,5]
        
        # save all variables to pandas (filtered results filtres)
        self._filtres = pd.DataFrame({'x':x, 'y':y, 'dx':dx, 'dy':dy, '2d2':d2, 'error':error})
        
    def TS_filt1(self):
        """
        combination of filters that are calibrated for the Moosfluh case study
        
        the error is adjusted to 2 if the result was filtered due to magnitude
        and adjusted to 3 if the results was filtered due to the error
        and adjusted to 4 if the results was filtered with the vector filter
        the error value can be used in the time series creation to identify 
        filtered points (e.g. magnitude filtered as rock falls)

        Returns
        -------
        temp : np.array
            filtered dic results.

        """
        # use a vector filter 
        temp = self.vector_filt()

        
        # set all results with an error larger than THR to nan and error to 3
        # temp[(temp[:,-1]>self.THR), 5] = 3
        # temp[(temp[:,-1] == 3), 2:5] = np.nan
        
        # get the filtered results of the threshold filter
        # self._filtout[(temp[:,-1]>self.THR)] = True
        
        
        #  set all results smaller than MAG to 0 and error to 2
        temp[(temp[:,-2]>self.MAG), 5] = 2
        temp[(temp[:,-1] == 2), 2:5] = np.nan
        
        # get the filtered results of the magnitude filter
        self._filtout[(temp[:,-2]>self.MAG)] = True
        
        return temp

    def thr_error_filt(self):
        """
        filters out all measurements with an error larger than THR

        Returns
        -------
        temp : np.array
            filtered DIC results.

        """
        
        # convert results to np.array
        temp = np.array(self._res)
        # set all results with an error larger than THR to nan
        temp[(temp[:,-1]<self.THR), :] = np.nan
        
        # a variable to save filtered locations
        self._filtout = np.zeros(self._res.shape[0],dtype=bool)
        self._filtout[np.isnan(temp[:,0])] = True
        
        return temp
        
    def thr_mag_filt(self):
        """
        filters out all measurements with a 2D magnitude larger than MAG

        Returns
        -------
        temp : np.array
            filtered DIC results.

        """
        
        # convert results to np.array
        temp = np.array(self._res,dtype=float)
        
        #  set all results smaller than MAG to nan
        temp[(temp[:,-2]>self.MAG), :] = np.nan
        
        # a variable to save filtered locations
        self._filtout = np.zeros(self._res.shape[0],dtype=bool)
        self._filtout[np.isnan(temp[:,0])] = True
        
        return temp
        
    def mean_filt(self):
        """
        smoothes the DIC results spatially with a 3x3 mean filter

        Returns
        -------
        rmf : np.array
            smoothed DIC results.

        """
        
        # convert DataFrame to np.array
        self._res = np.array(self._res)
        
        # get shapes for the matrix of the pixoff results
        size_tx = self._secondary.shape[1]
        size_ty = self._res.shape[0]
        anz_col = round(size_tx/(self._stepsize))
        
        # get dx
        t1 = self._res[:,2]
        # reshape to a matrix 
        mt1 = t1.reshape((int(t1.shape[0]/anz_col),int(anz_col))) 
        
        # get dy
        t2 = self._res[:,3] 
        # reshape to a matrix 
        mt2 = t2.reshape((int(t2.shape[0]/anz_col),int(anz_col)))
        
        # define a kernel which averages the pixels in a window of the size win
        kernell = np.full((self.MFWS, self.MFWS), 1/(self.MFWS**2)) 
        mtpostfilty = ndimage.convolve(mt2, kernell, mode='constant')
        mtpostfiltx = ndimage.convolve(mt1, kernell, mode='constant')
        
        # reshape filtered results to vector
        tpostfilty = mtpostfilty.reshape(size_ty,1)
        tpostfiltx = mtpostfiltx.reshape(size_ty,1)
        
        # combine filtered displacements with all result values
        rmf = np.concatenate((self._res[:,0], self._res[:,1], tpostfiltx[:,0], tpostfilty[:,0],
                              self._res[:,4], self._res[:,5])).reshape((-1, 6), order='F')
        return rmf
        
    def vector_filt(self):
        """
        Calculates the difference between the blurred (with nxn mean filter) 
        and original DX,DY, and magnitude fields and compares the differences
        to thresholds. If the thresholds are exceeded in any of the fields of
        differences, the considered measurement is filtered out.

        Returns
        -------
        tot_check : np.array
            filtered DIC results.

        """

        self._res = np.array(self._res)
        # max allowed difference between mean mag (3x3) and local mag deformation
        MAGCAP = self.WIN/self.VEC1
        # max allowed difference between mean x (3x3) and local x deformation
        XCAP = self.WIN/self.VEC2
        # max allowed difference between mean y (3x3)and local y deformation
        YCAP = self.WIN/self.VEC2
        
        # size of the mean filter kernell (e.g. 6 -> 6x6 window)
        n = 5

        # Preparation

        # try because shapes of different methods are not always the same
        try:
            # get the shape of the input images of the DIC
            size_tx = np.asarray(self._secondary.shape) 
            
            # get the shape of the DIC measurements in 2D
            size_tx = (math.floor(size_tx[0]/(self._stepsize))+1, math.floor(size_tx[1]/(self._stepsize))+1) 
            
            # shape of unfiltered results arranged in rows for later reshaping
            size_ty = self._res.shape[0]
            
            # define a kernel which averages the pixels in a window
            kernell = np.full((n, n), 1/(n**2)) 
            
            # get results DY, DX, Dtot (=magnitude) for DIC
            t1 = self._res[:,3] # DY
            t2 = self._res[:,2] # DX
            t3 = self._res[:,4] # Dtot
            
            # reshaping to 2D matrix of the results
            # such, the neighbours lay next to each other
            mt1 = np.array(t1.reshape(size_tx),dtype=float)
            mt2 = np.array(t2.reshape(size_tx),dtype=float)
            mt3 = np.array(t3.reshape(size_tx),dtype=float)
            
        except:
           
            # get the shape of the input images of the DIC
            size_tx = np.asarray(self._secondary.shape) 
            
            # get the shape of the DIC measurements in 2D
            size_tx = (math.floor(size_tx[0]/(self._stepsize))+1, math.floor(size_tx[1]/(self._stepsize))) 
            
            # shape of unfiltered results arranged in rows for later reshaping
            size_ty = self._res.shape[0]
            
            # define a kernel which averages the pixels in a window
            kernell = np.full((n, n), 1/(n**2)) 
            
            # get results DY, DX, Dtot (=magnitude) for DIC
            t1 = self._res[:,3] # DY
            t2 = self._res[:,2] # DX
            t3 = self._res[:,4] # Dtot
            
            # reshaping to 2D matrix of the results
            # such, the neighbours lay next to each other
            mt1 = np.array(t1.reshape(size_tx),dtype=float)
            mt2 = np.array(t2.reshape(size_tx),dtype=float)
            mt3 = np.array(t3.reshape(size_tx),dtype=float)
        
        # optinoal cut off if CUT larger than 0
        if self.CUT > 0:
            mt1[mt1>abs(self.WIN/self.CUT)] = np.nan
            mt2[mt2>abs(self.WIN/self.CUT)] = np.nan

        # Filtering
        
        # all steps done for DY,DX,Dtot
        
        # blur the results by using a mean filter
        imf1 = cv2.filter2D(mt1, -1, kernell)
        imf2 = cv2.filter2D(mt2, -1, kernell)
        imf3 = cv2.filter2D(mt3, -1, kernell)
        # reshape the mean values to the original shape of measurements per row
        imf1 = imf1.reshape(size_ty,1)
        imf2 = imf2.reshape(size_ty,1)
        imf3 = imf3.reshape(size_ty,1)
        # calculate the difference between mean value and local value
        abs1 = abs(t1-imf1[:,0])
        abs2 = abs(t2-imf2[:,0])
        abs3 = abs(t3-imf3[:,0])
        # set difference values larger than the threshold to nan
        abs1[abs1>YCAP] = np.NAN
        abs2[abs2>XCAP] = np.NAN
        abs3[abs3>MAGCAP] = np.NAN
        # create a matrix consisting of the original values and filtered ones 
        # all vectors are a column in the new matrix
        compare_values = np.array(np.concatenate((t1,t2,t3,abs1,abs2,abs3)).reshape((-1, 6), order='F'),dtype=float)
        # if 
        compare_values[np.isnan(compare_values).any(axis=1), :] = np.nan

        # # Vector direction
        
        # # to save the direction angle of the vector
        # direction1 = np.empty(mt1.shape)*np.nan
        
        # # get the angle from unit circle in rad, mt1 = dy, mt2 = dx
        # direction1[(mt2>0)&(mt1>0)] = np.arctan(mt1[(mt2>0)&(mt1>0)]/mt2[(mt2>0)&(mt1>0)])
        # direction1[(mt2<0)&(mt1>0)] = 3.141 - np.arctan(mt1[(mt2<0)&(mt1>0)]/abs(mt2[(mt2<0)&(mt1>0)]))
        # direction1[(mt2<0)&(mt1<0)] = 4.712 - np.arctan(mt2[(mt2<0)&(mt1<0)]/abs(mt1[(mt2<0)&(mt1<0)]))
        # direction1[(mt2>0)&(mt1<0)] = 4.712 + np.arctan(mt2[(mt2>0)&(mt1<0)]/abs(mt1[(mt2>0)&(mt1<0)]))
        
        # # get angles of vectors directing in a direction of a basis (x,y)
        # direction1[(mt2==0)&(mt1<0)] = 4.712
        # direction1[(mt2==0)&(mt1>0)] = 1.57
        # direction1[(mt2>0)&(mt1==0)] = 0
        # direction1[(mt2<0)&(mt1==0)] = 3.141
        
        # # neglect results with no displacement measured
        # direction1[(mt2==0)&(mt1==0)] = np.NaN
        
        # # to save the difference in direction of 3x3 neighbourhood
        # compare_dir = np.ones(mt1.shape)
        # for i in range(1,compare_dir.shape[0]-1):
        #     for j in range(1,compare_dir.shape[1]-1):
        #         mean_dir1 = direction1[i-1:i+2,j-1:j+2]
        #         mean_dir1 = np.nanmean(mean_dir1)
        #         compare_dir[i,j] = abs(direction1[i,j]-mean_dir1)
                
        # # filter when difference in direction exceeds a threshold
        # compare_dir[compare_dir>self.VEC3] = np.nan
        # # dont filter the results where no displacement was measured
        # compare_dir[(mt2==0)&(mt1==0)] = 1
        # # reshape the results from matrix to vector
        # compare_dir = compare_dir.T.reshape(size_ty,)
        # # combine values of compare_mag and filtered magnitude
        # compare_dir = np.array(np.concatenate((compare_values[2],compare_dir)).reshape((-1, 2), order='F'),dtype=float) 
        # # set whole row to NaN if one value is NaN in the same row
        # compare_dir[np.isnan(compare_dir).any(axis=1), :] = np.nan 
        # # redefine compare_values[2] newly
        # compare_values[2] = compare_dir[:,0]
        
        
        # Finalization & Output 
        # x,y,dx,dy,magn,error
        tot_check = np.array(np.concatenate((self._res[:,0],
                                             self._res[:,1],compare_values[:,1],compare_values[:,0],compare_values[:,2],
                                             self._res[:,5])).reshape((-1, 6), order='F'),dtype=float)
        
        # set the error value to 4 when the point was filtered
        tot_check[np.isnan(tot_check).any(axis=1), 5] = 4;
        
        # a variable to save filtered locations
        self._filtout = np.zeros(t1.shape,dtype=bool)
        self._filtout[np.isnan(tot_check[:,2])] = True
        
        return tot_check
    
    def createBG(self,s_name,path):
        """
        Saves one background image per day in a defined output folder

        Parameters
        ----------
        s_name : Path object
            path from pathlib to the secondary image

        path : Path object
            path to the output folder where the background images are saved.

        Returns
        -------
        None.

        """
        
        # extract the date of the secondary and change format of string
        date = s_name.stem.split('_')[-1]
        date = date[:4] + '-' + date[4:6] + '-' + date[6:8] + '.jpg'
        
        # create the path to the background image
        path0 = path.joinpath(date)
        
        # check if a BG image for this date already exists
        if not path0.exists():
            # if not, save the array to an grayscale image
            cv2.imwrite(str(path0),self._secondary)
    
    def plotter(self,path=None):
        """
        plots the filtered results of the DIC and saves them to the image folder

        Parameters
        ----------
        path : pathlib object, optional
            path to where the plot should be saved. It is also used to show 
            a text on the plot with the processing informations.

        Returns
        -------
        None.

        """

        # convert filtered results to np.array
        rr = np.array(self._filtres)
        
        # neglect all data points with a smaller magnitude than PLTMIN
        rr[(rr[:,-2]<self.PLTMIN), :] = np.nan
        
        # save results that were cut with the line above
        self._plotres = pd.DataFrame({'x':rr[:,1], 'y':rr[:,0], 'dx':rr[:,3], 'dy':rr[:,2],
                                      '2d2':(rr[:,3]**2+rr[:,2]**2)**0.5, 'error':rr[:,4]})

        # settings for the colorbar 
        aspect = 20
        pad_fraction = 0.5
        
        # get the maximum "real" displacement between the two images
        # do this by taking the tenth largest displacement 
        # (this means ignoring 9 measurements assumed to be outliers)
        if self.PLTMAX == None:
            tempmag = np.sort(np.array(self._plotres['2d2']))
            self.PLTMAX =  np.nanmax(tempmag[~np.isnan(tempmag)][:-10])
        
        # get the names of the images (dates of aquirement)
        if path != None:
            dates = Path(path).stem.split('-')[-2:]
            dates = 'primary:     ' + str(dates[-2]) + '\n' + 'secondary: ' + str(dates[-1])
        
        # plotting the results as colored points
        if self._plottype == 1:
            
            ##################################################################
            # start of the left plot, colorscale from 0 to PLTMAX 
            ##################################################################
            
            # create a figure with two plots next to each other 
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            
            # don't show the ticks and labels of the axes
            axes[0].axis('off')
            
            # plot the secondary image in the background
            im0 = axes[0].imshow(self._secondary,cmap='gray',alpha=0.3)
            
            # plot points with color according to magnitude
            q0 = axes[0].scatter(self._plotres['y'],self._plotres['x'],s=0.2,c=self._plotres['2d2'],
                             alpha=1,cmap='jet')
            
            # invert the y-axis direction to resemble to image
            plt.gca().invert_yaxis()
            
             # plot the filtered points (outliers, decorrelation) as magenta points
            if self._filtout.size > 1:
                allres = np.array(self._res)
                filtoutplt = allres[self._filtout,:]
                axes[0].plot(filtoutplt[:,0],filtoutplt[:,1],'.', ms=1, c= 'magenta',clim=(0,self.PLTMAX))
            
            # plot red circles around measurments that are of special interest
            axes[0].plot(1120,640,'o', mfc='none', ms=4, c= 'r')
            # axes[0].plot(1440,416,'o', mfc='none', ms=4, c= 'r')
            # axes[0].plot(736,256,'o', mfc='none', ms=4, c= 'r')
            # axes[0].plot(2336,1504,'o', mfc='none', ms=4, c= 'r')
            
            # add a text to the upper left (dates of image aquirements)
            if path != None:
                t = plt.text(0.01, 0.95, dates, size=5, color='black', transform=axes[0].transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
            
            # add a colorbar that fits the size of the plot
            divider = make_axes_locatable(axes[0])
            width = axes_size.AxesY(axes[0], aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.colorbar(q0, cax=cax, label='Deformation (pixels)')
        
            ##################################################################
            # start of the right plot, colorscale from 0 to MAG
            ##################################################################
            
            # don't show the ticks and labels of the axes
            axes[1].axis('off')
            
            # plot the secondary image in the background
            im1 = axes[1].imshow(self._secondary,cmap='gray',alpha=0.3)
            
            # plot points with color according to magnitude
            q1 = axes[1].scatter(self._plotres['y'],self._plotres['x'],s=0.2,c=self._plotres['2d2'],
                             alpha=1,cmap='jet')
            
            # invert the y-axis direction to resemble to image
            plt.gca().invert_yaxis()

             # plot the filtered points (outliers, decorrelation) as magenta points
            if self._filtout.size > 1:
                allres = np.array(self._res)
                filtoutplt = allres[self._filtout,:]
                axes[1].plot(filtoutplt[:,0],filtoutplt[:,1],'.', ms=1, c= 'magenta',clim=(0,self.MAG))
                
            # plot red circles around measurments that are of special interest
            axes[1].plot(1120,640,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(1440,416,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(736,256,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(2336,1504,'o', mfc='none', ms=4, c= 'r')
            
            # add a colorbar that fits the size of the plot
            divider = make_axes_locatable(axes[1])
            width = axes_size.AxesY(axes[1], aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.colorbar(q1, cax=cax, label='Deformation (pixels)')
            
            # change plot resolution
            plt.gcf().set_dpi(500)
                        
        # plot the filtered results as arrows with length and color according
        # to the magnitude of the 2D displacement
        if self._plottype == 2:
            
            ##################################################################
            # start of the left plot, colorscale from 0 to PLTMAX 
            ##################################################################
            
            # create a figure with two plots next to each other 
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            
            # don't show the ticks and labels of the axes
            axes[0].axis('off')
            
            # plot the secondary image in the background
            im0 = axes[0].imshow(self._secondary,cmap='gray',alpha=0.3)
            
            # plot arrows with length and color according to magnitude
            q0 = axes[0].quiver(list(self._plotres['y']),
                           list(self._plotres['x']),
                           list(self._plotres['dy']),
                           list(-self._plotres['dx']),
                           list(self._plotres['2d2']),
                            scale=None,width=0.01,units='inches', cmap='jet',clim=(0,self.PLTMAX))

            # add a colorbar that fits the size of the plot
            divider = make_axes_locatable(axes[0])
            width = axes_size.AxesY(axes[0], aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.colorbar(q0, cax=cax, label='Deformation (pixels)')
            
            # plot the filtered points (outliers, decorrelation) as magenta points
            if self._filtout.size > 1:
                allres = np.array(self._res)
                filtoutplt = allres[self._filtout,:]
                axes[0].plot(filtoutplt[:,0],filtoutplt[:,1],'.', ms=1, c= 'magenta')
                
            # plot red circles around measurments that are of special interest
            axes[1].plot(1120,640,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(1440,416,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(736,256,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(2336,1504,'o', mfc='none', ms=4, c= 'r')
            
            if path != None:
                t = plt.text(0.01, 0.95, dates, size=5, color='black', transform=axes[0].transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
            
            ##################################################################
            # start of the right plot, colorscale from 0 to MAG 
            ##################################################################
            
            # don't show the ticks and labels of the axes
            axes[1].axis('off')
            
            # plot the secondary image in the background
            im1 = axes[1].imshow(self._secondary,cmap='gray',alpha=0.3)
            
            # plot arrows with length and color according to magnitude
            q1 = axes[1].quiver(list(self._plotres['y']),
                           list(self._plotres['x']),
                           list(self._plotres['dy']),
                           list(-self._plotres['dx']),
                           list(self._plotres['2d2']),
                           scale=None,width=0.01,units='inches', cmap='jet',clim=(0,self.MAG))
            
            # plot the filtered points (outliers, decorrelation) as magenta points
            if self._filtout.size > 1:
                allres = np.array(self._res)
                filtoutplt = allres[self._filtout,:]
                axes[1].plot(filtoutplt[:,0],filtoutplt[:,1],'.', ms=1, c= 'magenta')
            
            # plot red circles around measurments that are of special interest
            axes[1].plot(1120,640,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(1440,416,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(736,256,'o', mfc='none', ms=4, c= 'r')
            # axes[1].plot(2336,1504,'o', mfc='none', ms=4, c= 'r')
            
            # add a colorbar that fits the size of the plot
            divider = make_axes_locatable(axes[1])
            width = axes_size.AxesY(axes[1], aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.colorbar(q1, cax=cax, label='Deformation (pixels)')
            
            # change plot resolution
            plt.gcf().set_dpi(500)
            
            
        if path != None:
            # save image if an output path is given when calling the method plotter
            plt.savefig(str(path)[:-4]+'.jpg')
            plt.show()
        else:
            plt.show()
            
        ######################################################################
        # following: old code that only plots one plot.
        
        # if self._plottype == 1:
        #     # plot the datapoints as points with color according to magnitude
        #     sc = plt.scatter(self._plotres['x'],self._plotres['y'],s=0.2,c=self._plotres['2d2'],
        #                      alpha=1,cmap='jet')
            
        #     # increase plot resolution
        #     plt.gcf().set_dpi(500)
            
        #     # create a colorbar
        #     plt.colorbar(sc, label='Deformation (pixels)')
            
        #     plt.plot(1440,416,'o', mfc='none', ms=4, c= 'r')
        #     plt.plot(736,256,'o', mfc='none', ms=4, c= 'r')
        #     plt.plot(2336,1504,'o', mfc='none', ms=4, c= 'r')
            
        
        # if self._plottype == 2:
        #     # plot arrows with length and color according to magnitude
        #     q = plt.quiver(list(self._plotres['x']),
        #                    list(self._plotres['y']),
        #                    list(self._plotres['dx']),
        #                    list(-self._plotres['dy']),
        #                    list(self._plotres['2d2']),
        #                    scale=None,width=0.01,units='inches', cmap='jet')
        #     # print(list(-self._plotres['dy']))
            
        #     # an arrow that has a standard size of 10 m
        #     # plt.quiverkey(q,0.9, 0.5, 10, r'$10 m$', labelpos='E',coordinates='figure')
            
        #     # create a colorbar
        #     plt.colorbar(q, label='Deformation (pixels)')
            
        #     # change limits of the axes
        #     plt.xlim(0, self._secondary.shape[1])
        #     plt.ylim(0, self._secondary.shape[0])
            
        #     # invert the y-axis direction to resemble to image
        #     plt.gca().invert_yaxis()
            
        #     # change plot resolution
        #     plt.gcf().set_dpi(500)
            
        #     plt.plot(1440,416,'o', mfc='none', ms=4, c= 'r')
        #     plt.plot(736,256,'o', mfc='none', ms=4, c= 'r')
        #     plt.plot(2336,1504,'o', mfc='none', ms=4, c= 'r')