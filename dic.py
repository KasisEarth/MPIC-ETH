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
from scipy.ndimage import shift
import numpy as np
import pygame
import math
import pandas as pd
from scipy.interpolate import interp2d

class PixOff(object):
    '''
    A class for co-registration and pixel offset (DIC) calculation from image pairs
    
    ...
    
    Attributes
    ----------
    OVSCO : int
        oversampling factor for co-registration
        
    SPLIT : int
        splitting image for co-registration, if = 1 no splitting
        
    TILESIZE : int
        size of the template in LAMMA (cxc, zncc)
    
    NODEDIST : int
        smallest grid size (level) in LAMMA (cxc, zncc)
        
    OVSLAMMA : int
        oversampling factor for LAMMA
        
    MAXBAND : array
        starting radius around template for cross correlation (search area)
        
    MAXSCALE : int
        maximal grid size in LAMMA (cxc, zncc)
        
    TOLERANCE : int
        tolerance which is added to the search radius in LAMMA
        
    WIN : int
        template size in FFT algorithm
        
    OVSFFT : int
        oversampling in FFT algorithm
        
    FFTSTEP : int
        step size in the fft algorithm (comparable to NODEDIST)
    
    Methods
    -------
    coregistration : co-registration of primary and secondary image, secondary image is shifted
    displayImage : displaying image to crop it for co-registration
    setup : set up for image cropping
    mainLoop : loop for image cropping
    cropImg : main method for image cropping
    dftups : plutin for FFT algorithm
    dftregistration : FFT algorithm
    pixelOffset : main method for the DIC, calling ethfft or lamma
    ethfft : main method for FFT, path preparation etc.
    lamma : main method for LAMMA 
    matching : ZNCC and CXC algorithm
    subpixOffsetMultiScale_nested: subpixel oversampling for lamma
    sgn: conversion from IN to OR image representation
    
    '''
    
    
    # input parameters of co-registration
    OVSCO = 4       # oversampling factor
    SPLIT = 1       # splitting of image into parts to register
    
    
    # input parameters of the LAMMA approach
    TILESIZE = 64                 # size of the template
    NODEDIST = 32                 # smallest grid size
    OVSLAMMA = 4                  # oversampling factor
    MAXBAND = [-20,20,-20,20]     # starting radius around template for cross correlation (search area)
    MAXSCALE = 256                # starting (maximal) scale
    TOLERANCE = 2                 # tolerance by which search area will be widened
    
    # input parameters for fft approach
    WIN = 64    # Window size [pix]
    OVSFFT = 4   # Oversampling factor
    FFTSTEP = 32 #step by which window is moved
    
    @property
    def results(self):
        '''
        returns the _res DataFrame (results of DIC)
        
        Returns
        -------
            pandas DataFrame
                DataFrame with columns x,y,dx,dy,d2,error/similarity value
        '''
        return self._res
    
    @property
    def secondary(self):
        
        return self._sec
    
    
    def __init__(self,primary,secondary):
        
        
        self._prim = primary
        self._sec = secondary
    
        self._cropind = 0 # cropping indices for co-registration, =0 if no crop
        
        self._res = 0 # results of the DIC, later defined as pandas DataFrame
    
    def coregistration(self,cropping=False,in_path='',cropindeces=0):
        '''
        
        Parameters
        ----------
        cropping : boolean, optional
            If True, the primary image will be displayed and an extent can
            be chosen with the mouse which will be used for co-registration.
            The default is False.
            
        in_path : string, optional
            path to the primary image (input folder).
            The default is ''.
            
        ovs : integer, optional
            factor by which the co-registration will be upsampled/oversampled.
            Upsample factor co_os 20 = images will be registered to within 1/20th of a pixel.
            The default is 1 (no upsamling)
            
        split : integer, optional
            splitting variable: if sp=1, entire image will be co-registrated as one, 
            sp > 1: the image will be split in even parts and every part will be co-registrated
            The default is 1 (no splitting of image).

        Returns
        -------
        None. registered images will be safed as _prim and _sec for later use

        '''

        
        # if cropping is 1, the cropping method is called to define the part
        # of the image for co-registration (interactively with mouse)
        if cropping == 1:
            self._cropind = self.cropImg(in_path)
            
        # if cropping is 2, the indeces in the main script are used to 
        # crop the image and use the cropped part for co-registration
        if cropping == 2:
            self._cropind = cropindeces
        
        # when cropping was 1 or 2, the images are cropped
        if self._cropind == 1 or self._cropind == 2:
            prim = self._prim[self._cropind[0]:self._cropind[1],self._cropind[2]:self._cropind[3]]
            sec = self._sec[self._cropind[0]:self._cropind[1],self._cropind[2]:self._cropind[3]]
        else:
            prim = self._prim
            sec = self._sec
            
        # when the split variable is larger than 1, the image is split into parts
        if self.SPLIT > 1:
            # make sure the split variable is an integer
            self.SPLIT = int(self.SPLIT)
            
            # find the dimensions of the splitted parts
            slice_length0 = prim.shape[0]//self.SPLIT
            slice_length1 = prim.shape[1]//self.SPLIT
            
            # empty lists to fill with indeces for splitting
            index0,index1 = [],[]
            shifted_list_row = []
            shifted_list_column = []
            
            # get the indeces for the splitted parts
            for j in range(self.SPLIT):
                index0.append(j*slice_length0)
                index1.append(j*slice_length1)
                
                # at the end, add the end index of the last splitted part
                if j == (self.SPLIT-1):
                    index0.append(prim.shape[0])
                    index1.append(prim.shape[1])
                    
            # loop through the splitted parts and co-register them with fft
            for i in range(self.SPLIT):
                for n in range(self.SPLIT):
                    zero,zero1 = index0[i],index0[i+1]
                    one,one1 = index1[n],index1[n+1]
                    
                    # Upsample factor OVSCO 20 = images will be registered to within 1/20th of a pixel.
                    # Default is 1 which means no upsampling. 
                    [_,_,shifty,shiftx] = self.dftregistration(np.fft.fft2(prim[zero:zero1,one:one1]),
                                                          np.fft.fft2(sec[zero:zero1,one:one1]),self.OVSCO)
                    
                    # append the shift to a list to calculate the mean shift later
                    shifted_list_row.append(shifty)
                    shifted_list_column.append(shiftx)
                    
            # calculate the mean shift of all splitted parts
            shifted = [sum(shifted_list_row)/len(shifted_list_row),sum(shifted_list_column)/len(shifted_list_column)]
            print('Detected Shift: ',shifted)

        # use entire images for co-registration when the split variable is <1
        if self.SPLIT <= 1:
            [_,_,shiftx,shifty] = self.dftregistration(np.fft.fft2(prim), np.fft.fft2(sec), self.OVSCO)
            shifted = [shiftx,shifty]
            print('Detected Shift: ',shifted)

        # save the secondary image after shifting it 
        self._sec = shift(self._sec, shift=(shifted[0], shifted[1]), mode='constant')
    

    

    def displayImage(self, screen, px, topleft, prior):
        """
        Enables visualization of primary and interactive definition of extent in image
        
        Source:
            Samplebias via Stackoverflow
            May 26 2011
            https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558

        """
        # ensure that the rect always has positive width, height
        
        x, y = topleft
        width =  pygame.mouse.get_pos()[0] - topleft[0]
        height = pygame.mouse.get_pos()[1] - topleft[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # eliminate redundant drawing cycles (when mouse isn't moving)
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # draw transparent box and blit it onto canvas
        screen.blit(px, px.get_rect())
        im = pygame.Surface((width, height))
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (x, y))
        pygame.display.flip()

        # return current box extents
        return (x, y, width, height)

    def setup(self, in_path):
        """
        Enables visualization of primary and interactive definition of extent in image
        
        Source:
            Samplebias via Stackoverflow
            May 26 2011
            https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558

        """
        
        px = pygame.image.load(in_path)
        screen = pygame.display.set_mode( px.get_rect()[2:] )
        screen.blit(px, px.get_rect())
        pygame.display.flip()
        return screen, px

    def mainLoop(self, screen, px):
        """
        Enables visualization of primary and interactive definition of extent in image
        
        Source:
            Samplebias via Stackoverflow
            May 26 2011
            https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558

        """
        
        topleft = bottomright = prior = None
        n=0
        while n!=1:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    if not topleft:
                        topleft = event.pos
                    else:
                        bottomright = event.pos
                        n=1
            if topleft:
                prior = self.displayImage(screen, px, topleft, prior)
        return ( topleft + bottomright )

    def cropImg(self, in_path):
        '''
        Enables visualization of primary and interactive definition of extent in image
        
        Source:
            Samplebias via Stackoverflow
            May 26 2011
            https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558

        Parameters
        ----------
        in_path : string
            path to the input folder to open the primary image with pygame.

        Returns
        -------
        list
            4x1 list with indices of the cropped extent in the primary image.

        '''
        pygame.init()
        screen, px = self.setup(in_path)
        left, upper, right, lower = self.mainLoop(screen, px)

        # ensure output rect always has positive width, height
        if right < left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower

        pygame.display.quit()

        return [upper,lower,left,right]

    
    def dftups(self,inpu,nor=0,noc=0,usfac=1,roff=0,coff=0):
        '''
        Upsampled DFT by matrix multiplication, can compute an upsampled DFT 
        in just a small region. 
        
        Parameters
        ----------
        inpu : np.array
            array that should be upsampled.
            
        nor,noc : int, optional
            Number of pixels in the output upsampled DFT, in
            units of upsampled pixels (default = size(inpu)).
            The default is 0.
            
        usfac : int, optional
            Upsampling factor. 
            The default is 1.
            
        roff,coff : float, optional
            Row and column offsets, allow to shift the output array to
            a region of interest on the DFT.
            The default is 0.
            

        Returns
        -------
        np.array: upsampled array
        
        Source
        ------
        
        Manuel Guizar - Dec 13, 2007
        Modified from dftus, by J.R. Fienup 7/31/06
        
        This code is intended to provide the same result as if the following
        operations were performed
          - Embed the array "in" in an array that is usfac times larger in each
            dimension. ifftshift to bring the center of the image to (1,1).
          - Take the FFT of the larger array
          - Extract an [nor, noc] region of the result. Starting with the 
            [roff+1 coff+1] element.
            
        It achieves this result by computing the DFT in the output array without
        the need to zeropad. Much faster and memory efficient than the
        zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]

        '''
    
        [nr,nc]=inpu.shape
        
        # Compute kernels and obtain DFT by matrix products
        kernc=np.exp((-1j*2*np.pi/(nc*usfac))*np.tile(( np.fft.ifftshift(np.linspace(0,nc-1,nc)).T - math.floor(nc/2) ),(noc,1)).T*( np.linspace(0,noc-1,noc) - coff ))
        kernr=(np.exp((-1j*2*np.pi/(nr*usfac))*( np.linspace(0,nor-1,nor) - roff )*np.tile(( np.fft.ifftshift(np.linspace(0,nr-1,nr)) - math.floor(nr/2)  ),(nor,1)).T)).T
        
        return np.dot(np.dot(kernr,inpu),kernc)
    
    
    def dftregistration(self,buf1ft,buf2ft,usfac=1): 
        '''
        Compute (upsampled) pixel shift between two arrays.

        Parameters
        ----------
        buf1ft : np.array
            Fourier transform of primary image, 
            DC in (1,1)   [DO NOT FFTSHIFT]
            
        buf2ft : np.array
            Fourier transform of secondary image, 
            DC in (1,1)   [DO NOT FFTSHIFT]
            
        usfac : integer, optional
            Upsampling factor (integer). Images will be registered to 
            within 1/usfac of a pixel. For example usfac = 20 means the
            images will be registered within 1/20 of a pixel.
            The default is 1.

        Returns
        -------
        output : tuple
            4x1 tuple with float elements:
                
                error: Translation invariant normalized RMS error between f and g
                
                diffphase: Global phase difference between the two images 
                           (should be zero if images are non-negative).
                           
                net_row_shift, net_col_shift: Pixel shifts between images

        '''

        if usfac == 0:
            CCmax = sum(sum(buf1ft[:,:]*np.conj(buf2ft)))
            rfzero = sum(abs(buf1ft[:,:])**2)
            rgzero = sum(abs(buf2ft[:,:])**2)
            error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
            error = np.sqrt(abs(error))
            diffphase=np.arctan2(np.imag(CCmax),np.real(CCmax))
            output=[error,diffphase]
                
        # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the peak
        elif usfac == 1:
            [m,n]=buf1ft.shape
            cc = np.fft.ifft2(buf1ft*np.conj(buf2ft))
            max1 = np.max(abs(cc))
            [locx,locy] = np.where(abs(cc)==max1)
            ccmax=cc[locx[0],locy[0]]
            rfzero = np.sum(abs(buf1ft)**2)/(m*n)
            rgzero = np.sum(abs(buf2ft)**2)/(m*n)
            error = 1.0 - ccmax*np.conj(ccmax)/(rgzero*rfzero)
            error = np.sqrt(abs(error))
            diffphase=np.arctan2(np.imag(ccmax),np.real(ccmax))
            md2 = np.fix(m/2)
            nd2 = np.fix(n/2)
            if min(locx) > md2:
                row_shift = min(locx) - m
            else:
                row_shift = min(locx)
            if min(locy) > nd2:
                col_shift = min(locy) - n
            else:
                col_shift = min(locy)
            output=[error,diffphase,row_shift,col_shift];
            
        # Partial-pixel shift
        else:
            # First upsample by a factor of 2 to obtain initial estimate
            # Embed Fourier data in a 2x larger array
            [m,n]=buf1ft.shape;
            mlarge=m*2
            nlarge=n*2
            cc=np.zeros((mlarge,nlarge),dtype = 'complex_');
            cc[int(m-np.fix(m/2)):int(m+1+np.fix((m-1)/2)),int(n-np.fix(n/2)):int(n+1+np.fix((n-1)/2))] = np.fft.fftshift(buf1ft)*np.conj(np.fft.fftshift(buf2ft))
          
            # Compute crosscorrelation and locate the peak 
            cc = np.fft.ifft2(np.fft.ifftshift(cc)) # Calculate cross-correlation
            max1 = np.max(abs(cc))
            [locx,locy] = np.where(abs(cc)==max1)
            ccmax=cc[locx[0],locy[0]]
            
            # Obtain shift in original pixel grid from the position of the
            # crosscorrelation peak 
            [m,n] = cc.shape
            md2 = np.fix(m/2)
            nd2 = np.fix(n/2)
            if min(locx) > md2:
                row_shift = min(locx) - m
            else:
                row_shift = min(locx)
            if min(locy) > nd2:
                col_shift = min(locy) - n
            else:
                col_shift = min(locy)
            row_shift=row_shift/2
            col_shift=col_shift/2
        
            # If upsampling > 2, then refine estimate with matrix multiply DFT
            if usfac > 2:
                ### DFT computation ###
                
                # Initial shift estimate in upsampled grid
                row_shift = np.round(row_shift*usfac)/usfac
                col_shift = np.round(col_shift*usfac)/usfac    
                dftshift = np.fix(math.ceil(usfac*1.5)/2) # Center of output array at dftshift+1
                
                # Matrix multiply DFT around the current shift estimate
                cc = np.conj(self.dftups(buf2ft*np.conj(buf1ft),math.ceil(usfac*1.5),math.ceil(usfac*1.5),usfac,dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac**2)
                                 
                # Locate maximum and map back to original pixel grid 
                max1 = np.max(abs(cc))
                [locx,locy] = np.where(abs(cc)==max1)
                ccmax=cc[locx[0],locy[0]]
                rg00 = self.dftups(buf1ft*np.conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2)
                rf00 = self.dftups(buf2ft*np.conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2)  
                locx = locx - dftshift
                locy = locy - dftshift
                row_shift = row_shift + locx/usfac
                col_shift = col_shift + locy/usfac
                row_shift = row_shift[0]
                col_shift = col_shift[0]
        
            # If upsampling = 2, no additional pixel shift refinement
            else:    
                rg00 = sum(sum( buf1ft*np.conj(buf1ft) ))/m/n
                rf00 = sum(sum( buf2ft*np.conj(buf2ft) ))/m/n
            error = 1.0 - ccmax*np.conj(ccmax)/(rg00*rf00)
            error = np.sqrt(abs(error))
            diffphase = np.arctan2(np.imag(ccmax),np.real(ccmax))
            
            # If its only one row or column the shift along that dimension has no effect. We set to zero.
            if md2 == 1:
                row_shift = 0
            if nd2 == 1:
                col_shift = 0
            output=[error,diffphase,row_shift,col_shift]
            
        return output
    
    def pixelOffset(self,method):
        '''
        main function for pixel offset calculation, calls DIC algorithm and
        saves the results to a panda DataFrame

        Parameters
        ----------
        method : string
            algorithm method, either fft, cxc, or zncc.

        Returns
        -------
        None.

        '''
        
        if method == 'fft':
            rr = self.ethfft(self._prim,self._sec)
        elif method == 'cxc' or method == 'zncc':
            rr = self.lamma(self._prim,self._sec,method)
        else:
            print('method not a valid string, change to fft or lamma')
            return # stop execution of function
            
        # get variables from rr array
        x = rr[:,1]
        y = rr[:,0]
        dx = rr[:,3] 
        dy = -rr[:,2]
        d2 = (dx[:]**2+dy[:]**2)**0.5
        error = np.array(rr[:,4],dtype=float)
        
        # save all variables to pandas
        self._res = pd.DataFrame({'x':x, 'y':y, 'dx':dx, 'dy':dy, '2d2':d2, 'error':error})

    
        
    def ethfft(self,An,Bn):
        """
        fast fourier transform DIC algorithm, based on Bickel et al. 2018
        
        Parameters
        ----------
        An : np.array
            primary image array.
        Bn : np.array 
            secoindary image array.

        Returns
        -------
        res : np.array
            DIC results: y,x,dy,dx,rmse.

        """

        # get dimensions of input image 
        [dimy,dimx] = An.shape
        
                
        # Padding (widen the image by half the window size)
        pad_width = int(self.WIN/2)
        An = np.pad(An,((pad_width,),(pad_width,)),mode='constant', constant_values=(0, 0))
        Bn = np.pad(Bn,((pad_width,),(pad_width,)),mode='constant', constant_values=(0, 0))
        
        # Get dimensions of the padded array
        [end_y,end_x] = An.shape 
        
        # Define dimensions of the result array
        res = np.empty((0,5))
        
        
        # Loop through images with the fft step size FFTSTEP
        # n = 0
        for i in range(int(self.WIN/2),int(end_y-self.WIN/2),int(self.FFTSTEP)):
            for j in range(int(self.WIN/2),int(end_x-self.WIN/2),int(self.FFTSTEP)):

                # crop chips of the images in the size of the template
                P0 = An[(i-int(self.WIN/2)):(i+int(self.WIN/2)+1),(j-int(self.WIN/2)):(j+int(self.WIN/2)+1)]
                P1 = Bn[(i-int(self.WIN/2)):(i+int(self.WIN/2)+1),(j-int(self.WIN/2)):(j+int(self.WIN/2)+1)]
                
                # register the chipfs after transfering to frequency domain
                [out0,out1,out2,out3] = self.dftregistration(np.fft.fft2(P0),np.fft.fft2(P1),self.OVSFFT)
                
                # results in coordinates
                # y-position / x-position / yoff(pix) / xoff(pix) / rmse
                res = np.vstack([res,[i-self.WIN/2, j-self.WIN/2, out2, -out3, out0]])
                
        # return the DIC results, will be rearranged in the PixOff method
        return res

    def lamma(self,primary,secondary,method='cxc'):
        """
        LAMMA DIC approach based on Dematteis et al. 2022

        Parameters
        ----------
        primary : np.array
            primary image array.
        secondary : np.array
            secondary image array.
        method : string, optional
            name of the algorithm used (cxc or zncc). The default is 'cxc'.

        Returns
        -------
        results : np.array
            DIC results, y,x,dy,dx,similarity measure.

        """
        
        # half the window size for later contemplation
        tileSz = math.floor(self.TILESIZE/2)
        
        # only regular grid is implemented in this routine
        regularGrid=True
        
        if regularGrid:
            
            # calculating the various spatial resolutions, start from the coarsest
            step = self.MAXSCALE
            vec = list()
            
            # get the grid sizes, from maxscale to nodedist
            while step >= self.NODEDIST:
                vec.append(int(step))
                step /= 2
                
            # sort the grids
            vec.sort()
            
            # get the number of scales
            numScales = len(vec)
            
            # determine the nodes of every scale
            [rw,cl]=primary.shape
            nodes = list()
            
            # get the grid nodes for every scale
            for ii in range(len(vec)-1,-1,-1):
                # step size of the considered scale
                step = vec[ii]
                
                # empty array to add the node indeces
                y_values = np.array([])
                x_values = np.array([])
                
                # get all the y indeces of each node in the grid of the considerd scale
                for kk in range(0,rw,step):
                    y_values = np.append(y_values, kk)
                # get all the x indeces of each node in the grid of the considerd scale
                for ww in range(0,cl,step):
                    x_values = np.append(x_values, ww)
                    
                # create a meshgrid for all the nodes and reshape indeces to vectors
                X,Y = np.meshgrid(y_values,x_values)
                X,Y = X.reshape([X.size,1]),Y.reshape([Y.size,1])
                
                # combine the indeces of the nodes in the grid
                twovec = np.concatenate((Y,X)).reshape((-1, 2), order='F')
                
                # add the indeces to a list, the list contains for each scale
                # an np.array with the two indeces 
                nodes.append(twovec)
                
            # exclude the nodes already present in previous levels
            n1 = nodes[0]
            for ii in range(1,len(nodes)):
                n2 = nodes[ii]
                for kk in range(len(n1)):
                    n2 = np.delete(n2, np.where((n2[:, 0] == n1[kk, 0]) & (n2[:, 1] == n1[kk, 1])), axis=0)
                nodes[ii]=n2
                n1 = np.concatenate((n1, n2), axis=0)
                
        # Determine the first neighbours of every node, considering the previous levels. 
        # The neighbours are used to adjust the search band limits in the    
        nody = list()
        for ii in range(len(nodes)-1,-1,-1):
            nody.append(nodes[ii])
        nodes = nody
        
        neighbour = list()
        for ii in range(0,numScales-1):
            nodesvv = list()
            for kk in range(ii+1,numScales):
                nodesvv.append(nodes[kk])
            dat1= np.vstack(nodesvv)
            dat1 = dat1[:,0]+1j*dat1[:,1]
            dat2 = nodes[ii]
            dat2 = dat2[:,0] + 1j * dat2[:,1]
            M = np.empty([len(dat2),4])
            for cc in range(len(dat2)):
                J = np.argpartition(abs(dat2[cc]-dat1), 3)[:4]
                M[cc,:] = J
            neighbour.append(M)
        neighbour.append([])
            
        DX = [np.nan] * numScales
        DY = [np.nan] * numScales
        NCC = [np.nan] * numScales
        # Loops on verious scales, starting from the coarsest
        for level in range(numScales-1,-1,-1):
            # indeces of all nodes of the current level/scale
            X = nodes[level][:,0]
            Y = nodes[level][:,1]
            if level == numScales-1:
                # in the first scale, the search band limits are those imposed by the user for all the nodes
                Xbm = np.ones((len(X),1))*self.MAXBAND[0]
                Xbp = np.ones((len(X),1))*self.MAXBAND[1]
                Ybm = np.ones((len(X),1))*self.MAXBAND[2]
                Ybp = np.ones((len(X),1))*self.MAXBAND[3]
        # ADAPTION --------------------------------------------------------------------
        # if information about displacements from larger grid is available, 
        # it is used to limit the search area in the current scale
            else:
                # YBM, YBP,XBM AND XBP ARE THE LIMITS OF THE SEARCH BAND IN THE DIRECTIONS DOWN, UP,LEFT, RIGHT
                # empty arrays where indeces for search area definition will be saved
                Xbm = [np.nan] * len(X)
                Xbp = [np.nan] * len(X)
                Ybm = [np.nan] * len(X)
                Ybp = [np.nan] * len(X)
                
                # get the information of the DIC in the larger scale
                tmpN = NCC[level+1:numScales]
                tmpN = [item for sublist in tmpN for item in sublist]
                tmpX = DX[level+1:numScales]
                tmpX = [item for sublist in tmpX for item in sublist]
                tmpY = DY[level+1:numScales]
                tmpY = [item for sublist in tmpY for item in sublist]

                for i in range(len(X)):
                    # get the neighbour's indeces of the current node i
                    # put them as integer into a list to asses the ncc,X,Y values of the neighbour nodes by indexing
                    idx = list()
                    
                    # fill the index list idx with the neighbours
                    for nn in range(4):
                        idx.append(int(neighbour[level][i,nn]))
                        
                    # when all similarity measures at the neighbour positions of the
                    # larger scale are nan, the maxbands are used again
                    if np.all(np.isnan(np.array(tmpN)[idx])):
                        Ybm[i]=np.array(self.MAXBAND[2])
                        Ybp[i]=np.array(self.MAXBAND[3])
                        Xbm[i]=np.array(self.MAXBAND[0])
                        Xbp[i]=np.array(self.MAXBAND[1])

                    else:
                        # if there are more than 2 neighbours, sort them with 
                        # increasing ncc values and get rid of the one or two
                        # nodes with the worst similarity measures
                        if np.sum(~np.isnan(np.array(tmpN)[idx]))>2:
                            idx=np.array(idx)[~np.isnan(np.array(tmpN)[idx])]
                            reliable=np.argsort(np.array(tmpN)[idx])
                            if np.sum(~np.isnan(np.array(tmpN)[idx]))>3:
                                idx = np.array(idx)[reliable[2:3]]
                            else:
                                idx = idx[reliable[1:2]]
                            
                        # define the band limits with the neighbour nodes with largest ncc vlaues
                        # add the self.TOLERANCE term to the band limits
                        Ybm[i] = np.nanmin(np.array(tmpY,dtype=float)[idx])- self.TOLERANCE
                        Ybp[i] = np.nanmax(np.array(tmpY,dtype=float)[idx])+ self.TOLERANCE
                        Xbm[i] = np.nanmin(np.array(tmpX,dtype=float)[idx])- self.TOLERANCE
                        Xbp[i] = np.nanmax(np.array(tmpX,dtype=float)[idx])+ self.TOLERANCE

        # ADATPTION ENDS ---------------------------------------------------------------
        
            # reset the variables that will be filled with the DIC results
            dx = [np.nan]*len(X)
            dy = [np.nan]*len(X)
            ncc = [np.nan]*len(X)
            void = [np.nan]*len(X)
            
            for jj in range(len(X)):
                # get the index of the current node
                x0 = X[jj]
                y0 = Y[jj]
                
                # take the search area limits for the current node
                try:
                    rm = round(float(Ybm[jj]))
                except:
                    rm = self.MAXBAND[0]
                try:
                    rp = round(float(Ybp[jj]))
                except:
                    rp = self.MAXBAND[1]
                try:
                    cm = round(float(Xbm[jj]))
                except:
                    cm = self.MAXBAND[2]
                try:
                    cp = round(float(Xbp[jj]))
                except:
                    cp = self.MAXBAND[3]
                    
                # check if the search band goes outside the image. if it does, the image matching results are set to NaN
                tol = self.TOLERANCE *(numScales-level+1)+1
                cond1 = (y0-tileSz+self.MAXBAND[2]-tol < 1)
                cond2 = (y0+tileSz+self.MAXBAND[3]+tol > secondary.shape[0])
                cond3 = (x0-tileSz+self.MAXBAND[0]-tol < 1)
                cond4 = (x0+tileSz+self.MAXBAND[1]+tol > secondary.shape[1])
                if cond1 or cond2 or cond3 or cond4:
                    dx[jj] = np.nan
                    dy[jj] = np.nan
                    ncc[jj] = np.nan
                    void[jj] = 1
                
                # if search band does not go out, crop out the reference and
                # search patch to prepare for the matching
                else:
                    void[jj] = 0
                    
                    # get reference patch
                    refTile = primary[int(y0-tileSz):int(y0+tileSz+1),
                                      int(x0-tileSz):int(x0+tileSz+1)].astype(float)
                    
                    # get secondary patch, limits depend on the iterrogation area limits
                    searchTile = secondary[int(y0-tileSz+rm):int(y0+tileSz+rp+1),
                                           int(x0-tileSz+cm):int(x0+tileSz+cp+1)].astype(float)
                    
                    # check if the patches are not Nan and if they have not constant values. if the case, set results to Nan
                    if np.isnan(np.min(refTile)) or np.isnan(np.min(searchTile)) or len(np.unique(refTile))==1 or len(np.unique(searchTile))==1:
                        dx[jj] = np.nan
                        dy[jj] = np.nan
                        ncc[jj] = np.nan
                        void[jj] = 1
                    else:
        # MATCHING -------------------------------------------------------------------
                        #calcualte the similarity function
                        [dx[jj],dy[jj],DCC] = self.matching(refTile, searchTile, [cm,cp,rm,rp], self.OVSLAMMA, method)
                        
                        # get the results out of a np.array and make it a float
                        try:
                            dx[jj] = float(dx[jj])
                            dy[jj] = float(dy[jj])
                            ncc[jj] = DCC

                        except:
                            dx[jj] = np.nan
                            dy[jj] = np.nan
                            ncc[jj] = np.nan
                            
            # store the vectors dx,dy,ncc in a list 
            # each element of the list represents a level/scale
            DX[level] = dx
            DY[level] = dy
            NCC[level] = ncc
        
        # combine the results of several grid levels
        res_x = nodes[0][:,1]
        for ii in range(1,len(nodes)):
            res_x = np.append(res_x,nodes[ii][:,1])
            res_x.reshape((res_x.shape[0],))
        res_y = nodes[0][:,0]
        for ii in range(1,len(nodes)):
            res_y = np.append(res_y,nodes[ii][:,0])
            res_y.reshape((res_y.shape[0],))
        res_dx = DX[0]
        for ii in range(1,len(nodes)):
            res_dx = np.append(res_dx,DX[ii])
            res_dx.reshape((res_dx.shape[0],))
        res_dy = DY[0]
        for ii in range(1,len(nodes)):
            res_dy = np.append(res_dy,DY[ii])
            res_dy.reshape((res_dy.shape[0],))
        res_ncc = NCC[0]
        for ii in range(1,len(nodes)):
            res_ncc = np.append(res_ncc,NCC[ii])
            res_ncc.reshape((res_ncc.shape[0],))

        # put the results of all scales into a np.array like in the fft approach
        # naming might be wrong, inputs should be: y, x, dy, dx, similarity measure
        # sorry for the confusion
        results = np.concatenate((res_x,res_y,-res_dy,res_dx,res_ncc)).reshape((-1, 5), order='F')
        return results

    
    def subpixOffsetMultiScale_nested(self,DCC,searchBand,os):
        """
        

        Parameters
        ----------
        DCC : array of float
            array of similarity measure which should be oversampled.
        searchBand : array of float
            array with the number of pixels in each direction, 
            gives size of the search window.
        os : integer
            oversampling factor.

        Returns
        -------
        float, float
            oversampled shift of the template in pixel coordinates.

        """
        
        # to navigate where the refernce frame was situated
        RowOffset = searchBand[2]
        ColOffset = searchBand[0]
        
        maxvalue = np.max(DCC)
        [I,J] = np.where(DCC==maxvalue)
        try:
            FirstApproxRow = float(I)+RowOffset
            FirstApproxCol = float(J)+ColOffset
        except:
            return np.nan,np.nan
        #pad array that we dont index out of dcc boundary
        DCCnan = np.pad(DCC,2,mode='constant',constant_values=(np.nan,))
        InterpArea = DCCnan[int(I):int(I+5),int(J):int(J+5)]
        if np.all(~np.isnan(InterpArea)):
            #create meshgrid from -1 to 1 (indeces of 3x3 area)
            xvalues,yvalues = list(np.linspace(-2,2,5)),list(np.linspace(-2,2,5))
            JJ,II = np.meshgrid(yvalues,xvalues)
            sizeos = 1+4*os
            
            #interpolate/oversample the 3x3 area
            f = interp2d(xvalues, yvalues, InterpArea, kind='cubic')
            
            #create meshgid from -os to os (indeces for interpolated/oversampled area)
            xvalues,yvalues = list(np.linspace(-2,2,sizeos)),list(np.linspace(-2,2,sizeos))
            nJJ,nII = np.meshgrid(yvalues,xvalues)
            output = f(xvalues, yvalues)
            
            # output = cv2.resize(InterpArea, (sizeos,sizeos), interpolation=cv2.INTER_CUBIC)
            # get max in ovesampled patch, this is the decimal point offset
            maxvalue = np.max(output)
            pos = np.where(output.flatten('F')==maxvalue)
            subdx = nII.flatten('F')[pos]
            subdy = nJJ.flatten('F')[pos]
        else:
            subdx,subdy = 0,0

        return FirstApproxCol+subdy,FirstApproxRow+subdx
            
        
    
    def matching(self,refTile,searchTile,searchBand,os=1,method='cxc'):
        """
        

        Parameters
        ----------
        refTile : np.array
            MxN reference patch
            
        searchTile : np.array
            PxR reference patch P>=M; R>=N
            
        searchBand : np.array
            1x4 positive integer array. Width of the search band in
            the directions [left right up down]
            
        os : int, optional
            numerical value of the subpixel sensitivity.
            if os = 1: no oversampling. The default is 1.
            
        method : string, optional
            String that indicates the similarity function. zncc or cxc.
            The default is 'cxc'.

        Returns
        -------
        DX : float
            rightward motion of the refTile (px)
        DY : float
            downward motion of the refTile (px)
        DCC : float
            max value of DCC

        """

        # get the size of the searchBand
        [cm,cp,rm,rp] = searchBand
        
        # create a meshgrid with the indexes of all positions the reference 
        # patch can have within the search patch
        y_values = np.array([])
        x_values = np.array([])
        for kk in range(cm,cp+1):
            y_values = np.append(y_values, kk)
        for ww in range(rm,rp+1):
            x_values = np.append(x_values, ww)
        X,Y = np.meshgrid(y_values,x_values)
        dcc = np.zeros(X.shape)
        [refRow,refCol] = refTile.shape
        [searchRow, searchCol] = searchTile.shape
        
        # convert to OR representation for CXC algorithm
        if method == 'cxc':
            A = self.sgn(refTile)
            searchTile = self.sgn(searchTile)
        if method == 'zncc':
            A = refTile
            A = A-np.nanmean(A)
            B = np.nansum(A**2)
            
        # calculate at every posible position the ZNCC or CXC similarity measure
        for ii in range(dcc.shape[0]):
            for jj in range(dcc.shape[1]):
                C = searchTile[ii:ii+refRow,jj:jj+refCol]
                if method == 'zncc':
                    dcc[ii,jj] = np.nansum(A*(C-np.nanmean(Y))) / np.sqrt(B*np.nansum((C-np.nanmean(C))**2))
                if method == 'cxc':
                    dcc[ii,jj] = np.nanmean(np.real(np.conj(A)*C))
                
        if os>1:
            # compute subpixel displacement
            DX,DY = self.subpixOffsetMultiScale_nested(dcc,searchBand,os)
            maxvalue = np.max(dcc)
            # print(DX,DY)
            
        if os==1:
            #identify the shift position
            maxvalue = np.max(dcc)
            pos = np.where(dcc.flatten('F')==maxvalue)
            if pos[0].size > 1:
                DX,DY = np.nan,np.nan 
            else:
                DX = X.flatten('F')[pos]
                DY = Y.flatten('F')[pos]

        return DX,DY,maxvalue
        
    
    
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
        
        # However the following is implemented  in LAMMA routine (Dematteis & Gordian 2022)
        # normalize array of complex numbers
        # or_array = x/np.abs(x)
        
        return or_array
    
    
if __name__ == '__main__':
    import cv2
    
    primary = cv2.imread('00test001.jpg', cv2.IMREAD_GRAYSCALE)
    secondary = cv2.imread('00test002.jpg', cv2.IMREAD_GRAYSCALE)
    hoi = PixOff(primary,secondary)
    test_results = hoi.dftregistration(np.fft.fft2(primary),np.fft.fft2(secondary),2)
    
    
