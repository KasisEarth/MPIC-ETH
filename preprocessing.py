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
import numpy as np
from scipy.ndimage.filters import uniform_filter
import scipy.ndimage

class PreProcessing(object):
    '''
    Class for image preprocessing (wallis filter and OR representation) before DIC analysis
    
    Can be initialized with four inputs (two uint8 np.arrays, a boolean, and a string)
        to automatically start preprocessing with these images.
        
    Or can be initialized with no inputs to use its methods with external images.
        For example, call: 
            prepro = PreProcessing()
            prepro.wallis(image_name)
    
    Attributes
    ----------
    WIN : int (even number)
        block size of the wallis filter, standard: 32px 
        
    TARM : int
        target mean value in a block after wallis filtering, 
        standard: 150
        
    TARS : int
        target standard deviation in block after wallis filtering, 
        standard: 150
        
    BRIGHTNESS : float
        brightness enforcing constant of wallis filter
        
    CONTRAST : float
        contrast enforcing constant of wallis filter
    


    Properties
    ----------
    primary : np.array
        first image of the image pair of the digital image correlation
        
    secondary : np.array
        second image of the image pair of the digital image correlation
    '''
    
    # Definition of wallis paramters 
    WIN = 32                # block size (even number)
    TARM = 150              # target mean
    TARS = 150              # target standard deviation
    BRIGHTNESS = 1.0        # brightness enforcing constant
    CONTRAST = 0.9995       # contrast enforcing constant
    
    def primary(self):
        '''
        returns the _prim array (call after using wallis() or sgn())
        
        Returns
        -------
            np.array
                Array of the primary image, potentially processed with 
                wallis filter and/or in OR representation
        '''
        return self._prim
    
    def secondary(self):
        '''
        returns the _sec array (call after initiation when wallis/sgn was run)
        
        Returns
        -------
            np.array
                Array of the secondary image, potentially processed with 
                wallis filter and/or in OR representation
        '''
        return self._sec
    
    
    def __init__(self, primary=None, secondary=None, wallis=False, representation='in'):
        ''' 
        
        Constructor of class PreProcessing
        
        Initiates wallis filter and/or convertion to OR representation if all four
            input parameters are given accordingly (wallis=True or representation='or')
        
        Parameters
        ----------
        primary,secondary : np.array
            numpy arrays of the first and second image in the DIC
        
        wallis : boolean
            If True, wallis filter will be applied
            
        representation : string
            If = 'or', the representation of the input images will be changed
            to orientation (originally intensity representation)
        
        
        '''
        # get the primary and secondary image arrays
        self._prim = primary
        self._sec = secondary
        
        # get the boolean wallis, later run wallis if wal == True
        self._wal = wallis
        
        # get the boolean represenation, later run sgn() if _repr == 'or'
        self._repr = representation

        if self._wal:
            # use a wallis filter on the input images
            self._prim = self.wallis(self._prim)
            self._sec  = self.wallis(self._sec)
            
        if self._repr == 'or':
            # change representation of image from intensity 'in' to orientation 'or'
            self._prim = self.sgn(self._prim)
            self._sec  = self.sgn(self._sec)
            
            
    def wallis(self,img_array):
        '''
        
        Filter that adjusts the intensity mean and deviation within a block size to a given value
        Parameters are defined as Attributes of the class (WIN,TARM,TARS,BRIGHTNESS,CONTRAST)

        Parameters
        ----------
        img_array : np.array (uint8)
            input array of an image that should be filtered with a wallis filter

        Returns
        -------
        filtered_array : np.array
            array of the input image filtered with a wallis filter


        '''
        if self.WIN%2==0:
            self.WIN +=1 #adjust the window size if the number is not even.
        
        # convert to float array from 0-1
        img1 = img_array.astype('float') / (255)

        
        # Padding (widen the image by half the window size)
        pad_width = int(self.WIN/2)
        im1 = np.pad(img1,((pad_width,),(pad_width,)),mode='symmetric')
        
        # Get dimensions of both images/arrays
        dim1 = im1.shape;
        
        # define a kernel which averages the pixels in a window of the size win
        kernell = np.full((self.WIN+1, self.WIN+1), 1/((self.WIN+1)**2)) 
        # Blur the images, equivalent to imfilter in matlab
        imf1 = scipy.ndimage.correlate(im1, kernell, mode='constant')
        
        
        # Use a standard filter
        imstd1 = self.window_stdev(im1)*np.sqrt((self.WIN**2)/((self.WIN-1)**2))
        
        # Calculate wallis filter
        imG1 = (im1[:,:]-imf1[:,:])*self.CONTRAST*self.TARS/(self.CONTRAST*imstd1[:,:]+(1-self.CONTRAST)*self.TARS)
        +self.BRIGHTNESS*self.TARM+(1-self.BRIGHTNESS)*imf1[:,:]
        
        # cut values above 255 and below 0
        imG1[imG1 < 0] = 0
        imG1[imG1 > 255] = 255
        
        # De-Padding and conversion to uint8 format
        top = int(self.WIN/2-1)
        filtered_array = imG1[top:int(dim1[0]-self.WIN/2),top:int(dim1[1]-self.WIN/2)].astype(np.uint8)

        return filtered_array

    def window_stdev(self,X):
        '''
        
        Parameters
        ----------
        X : np.array
            array that should be treated with a standard filter.

        Returns
        -------
        filtered : np.array
            array filtered with a standard filter.

        '''
        c1 = uniform_filter(X, self.WIN, mode='reflect')
        c2 = uniform_filter(X*X, self.WIN, mode='reflect')
        filtered = np.sqrt(abs(c2 - c1*c1))
        return filtered
        
        
    def sgn(self,img_array):
        '''
        
        Parameters
        ----------
        img_array : np.array
            input array in Intensity representation (IR) that should be 
            converted to Orientation representation (OR).

        Returns
        -------
        or_array : np.array
            Output array that is now in OR representation.

        '''
        # create gradient in x and y direction of the array
        grad = np.gradient(np.array(img_array,dtype=float))
        
        # combine gradients into a complex number
        x = grad[1]+grad[0]*1j
        
        # normalize numbers only larger than 0
        # This is suggested in (Dematteis & Gordian 2021)
        x[x!=0] = x[(x!=0)] / np.absolute(x[(x!=0)])
        x[x==0] = 0
        or_array = x
        
        # However this is implemented  in LAMMA routine (Dematteis & Gordian 2022)
        # normalize array of complex numbers
        # or_array = x/np.abs(x)
        
        return or_array
        
        
        
        
        