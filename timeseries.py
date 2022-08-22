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

from pathlib import Path     
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from pylab import cm
import cv2          # module for image reading and image manipulation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import copy
import imageio

class TimeSeries(object):
    '''
    Class for creation of time series
    
    Attributes
    ----------
    HOURS : int
        the number of hours needed in a day to consider it within the time period
        
    SINPLEPLOT : boolean
        if True, a single plot for the gif is created
        
    MULTIPLOT : boolean
        if True, two plots for the gif are created
        
    scale_2d : list
        list of minimum and maximum of the scale for the 2D GIF creation    
        [min velocity, max velocity, min cumulative displacement, max cum. dip.]

    scale_3d : list
        list of minimum and maximum of the scale for the 3D GIF creation    
        [min velocity, max velocity, min cumulative displacement, max cum. dip.]

    '''
    # the number of hours needed in a day to consider it within the time period
    HOURS = 12
    
    # settings for the GIF creation
    SINGLEPLOT = False
    MULTIPLOT = True
    
    # minimum and maximum of scale for the GIF creation
    # [min velocity, max velocity, min cumulative displacement, max cum. dip.]
    scale_3d = [0.25,5,3,15]
    scale_2d = [0.3,6,3,30]
    
    def __init__(self,inpath,tsfilt=False):
        """
        Inititation function of the TimeSeries Class

        Parameters
        ----------
        inpath : object of the pathlib
            path to the georectified 3D results of the DIC.

        Returns
        -------
        None.

        """
        # path to the input folder for the time series creation 
        # usually, this is the path to the output folder of the DIC
        self._inpath = Path(inpath)

        # get all names of the csv files in the input folder
        self._input_list = [p for p in self._inpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).startswith('filt')]

        # path to output folder
        self._outpath = self._inpath.joinpath("time-series")
        
        # path to hourly velocities
        self._hourpath = self._outpath.joinpath("hours")
        
        # dictionary where the average results per day are saved
        self._datedict = {}

        # create the output folder in the input folder if it is not existent
        self._outpath.mkdir(parents=True, exist_ok=True)
        self._hourpath.mkdir(parents=True, exist_ok=True)
        
        # if True, a time series filter will be applied
        self._ts_filt = tsfilt


    def create_ts(self):
        """
        Averages all displacement results per hour.
        
        In Detail:
            Takes all 2d results from the inpath and calculates the displacement
            per hour. Saves the hourly velocity in the hour folder. Calculates
            the average hourly velocity per day and saves it in the time series
            folder. Calculates the cumulated displacements and saves them in the
            time series folder
    
        Returns
        -------
        None.
    
        """
        
        print('--- Calculate velocity per hour')
        for fpath in self._input_list:
            # print(str(fpath))
            # import the results with path fpath
            try:
                res = np.genfromtxt(str(fpath), delimiter=',')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            except:
                res = np.genfromtxt(str(fpath), delimiter=';')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
        
        
            # extract start and enddate from the name of the file
            # name of the file should end with the startdate-enddate
            # date in name with format: JJJJMMDDTHHMMSS
            startdate = fpath.stem.split('-')[-2].split('T')[0][-8:]
            enddate = fpath.stem.split('-')[-1].split('T')[0][-8:]
            
            # extract time in int fromat. maybe will be used to define days that were really considered
            starttime = fpath.stem.split('-')[-2].split('T')[1]
            endtime = fpath.stem.split('-')[-1].split('T')[1]
            
            datetime0 = datetime(int(startdate[0:4]),int(startdate[4:6]),int(startdate[6:]),
                             int(starttime[:2]),int(starttime[2:4]),int(starttime[4:]))
            datetime1 = datetime(int(enddate[0:4]),int(enddate[4:6]),int(enddate[6:]),
                             int(endtime[:2]),int(endtime[2:4]),int(endtime[4:]))
            
        
        
        
            
            if datetime0.minute > 30:
                datetime0 = datetime(datetime0.year,datetime0.month,datetime0.day,
                                      datetime0.hour+1,00,00)
            else:
                datetime0 = datetime(datetime0.year,datetime0.month,datetime0.day,
                                      datetime0.hour,00,00)
                
            # round datetime1 to one hour and add one hour to include the last
            # hour in the while loop
            if datetime1.minute > 30:
                datetime1 = datetime(datetime1.year,datetime1.month,datetime1.day,
                                      datetime1.hour+2,00,00)
            else:
                datetime1 = datetime(datetime1.year,datetime1.month,datetime1.day,
                                      datetime1.hour+1,00,00)
                
            timediff = datetime1-datetime0
            hourdiff = int(timediff.days*24+timediff.seconds/3600)
            
            # dictionary to safe the number of measurements per hour
            hourdict = {}
        
            
            # get the displacements in 2d from the filtered DIC results
            res2d = res[:,2:4]
            # res2d[np.isnan(res2d)] = 0
            
            # caluclate from displacement to velocity (= disp/nr of hours)
            res2d = res2d/hourdiff
            
            # define variable date to loop through all hours within the baseline
            vardate = datetime0
            
            while vardate != datetime1:
            
                # create export name
                exname = str(vardate).replace(':','-') + '.csv'
                    
                if str(vardate) not in hourdict:
                    
                    # if the hour considered has not been saved yet, a new entry in the
                    # date dictionary will be added and a file saved into the hours folder
                    hourdict[str(vardate)] = 1
                    
                    # create velocity array
                    # columns: u,v,du,dv
                    velarray = np.empty((res.shape[0],5))*np.nan
                    velarray[:,0:2] = res[:,0:2]
                    velarray[:,2:4] = res2d
                    velarray[:,4] = res[:,5]
                    
                    
                    velo = pd.DataFrame({'start u [px]':velarray[:,0], 'start v [px]':velarray[:,1],
                                         'du [px/h]':velarray[:,2], 'dv [px/h]':velarray[:,3],
                                         'max error':velarray[:,4]})
                    # save the resulting pandas DataFrame to csv
                    velo.to_csv(str(self._hourpath.joinpath(exname)),index=False)
                    
                    # np.savetxt(self._hourpath.joinpath(str(vardate)+".csv"), velarray, delimiter=",")
                    
                else:
                    
                    try:
                        velarray = np.genfromtxt(str(self._hourpath.joinpath(str(vardate)+".csv")), delimiter=',')[1:,:]
                    except:
                        velarray = np.genfromtxt(str(self._hourpath.joinpath(str(vardate)+".csv")), delimiter=';')[1:,:]
                    
                    velarray[~np.isnan(res2d[:,0]),2:4] = (velarray[~np.isnan(res2d[:,0]),2:4]*hourdict[str(vardate)]+res2d[~np.isnan(res2d[:,0]),:])
                    velarray[~np.isnan(res2d[:,0]),2:4] = velarray[~np.isnan(res2d[:,0]),2:4]/(hourdict[str(vardate)] + 1)
        
                    # replace the start location if it was nan before
                    # first get the coordinates from the self._datedict dicitonary...
                    avu = velarray[:,0]
                    avv = velarray[:,1]
                    # then get the coordinates of the now considered results
                    u = res[:,0]
                    v = res[:,1]
        
                    # replace average coordinates where they are nan
                    avu[np.isnan(avu)] = u[np.isnan(avu)]
                    avv[np.isnan(avv)] = v[np.isnan(avv)]
        
                    # and finally replace the coordinates in the dictionary
                    velarray[:,0] = avu
                    velarray[:,1] = avv
        
                    # add 1 to the number of considered measurements for this hour
                    hourdict[str(vardate)] = hourdict[str(vardate)] + 1
                    
                    # set the min error to the error of the considered measurement
                    # if the error is smaller. Like this, the decorrelation is only 
                    # in the velocities that really are decorrelated
                    velarray[velarray[:,4]>res[:,5],4] = res[velarray[:,4]>res[:,5],5]
                    
                    velo = pd.DataFrame({'start u [px]':velarray[:,0], 'start v [px]':velarray[:,1],
                                         'du [px/h]':velarray[:,2], 'dv [px/h]':velarray[:,3],
                                         'min error':velarray[:,4]})
                    
                    # save the resulting pandas DataFrame to csv
                    velo.to_csv(str(self._hourpath.joinpath(exname)),index=False)
                    
                    # np.savetxt(str(self._hourpath.joinpath(str(vardate)+".csv")), velarray, delimiter=",")
                    
                vardate = vardate + timedelta(hours = 1)
                
                    
        print('--- Calculate average velocity per hour for each day')   
        # dictionary to save the number of considered day results
        datedict = {}
        
        # list of all hour files created in the hours folder
        hour_list = [p for p in self._hourpath.iterdir() if p.suffix.lower() == '.csv']
        
        for fname in hour_list:
            try:
                res = np.genfromtxt(str(fname), delimiter=',')[1:,:]
            except:
                res = np.genfromtxt(str(fname), delimiter=';')[1:,:]
                
            if fname.stem[:10] not in datedict:
                datedict[fname.stem[:10]] = [1,0]
                
                # create an array that is =1 when the error is large (means 
                # that filtering is applied)
                large_error = res[:,4]>=1
                large_error = np.array(large_error,dtype=int)
                datedict[fname.stem[:10]][1] = large_error
                
                velo = pd.DataFrame({'start u [px]':res[:,0], 'start v [px]':res[:,1],
                                     'daily average du [px/h]':res[:,2], 'daily average dv [px/h]':res[:,3],
                                     'max error':res[:,4]})
                # save the resulting pandas DataFrame to csv
                velo.to_csv(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')),index=False)
                
                # np.savetxt(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')), res, delimiter=",")
            else:
                try:
                    velarray = np.genfromtxt(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')), delimiter=',')[1:,:]
                except:
                    velarray = np.genfromtxt(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')), delimiter=';')[1:,:]
                
                # include the displacements of the new results in the velocity
                # per hour daily to get an average value there
                velarray[~np.isnan(res[:,2]),2:4] = (velarray[~np.isnan(res[:,2]),2:4]*datedict[fname.stem[:10]][0]+res[~np.isnan(res[:,2]),2:4])/(datedict[fname.stem[:10]][0]+1)
                
                # add to the array in the date dict when the error is large
                large_error = res[:,4]>=1
                large_error = np.array(large_error,dtype=int)
                datedict[fname.stem[:10]][1] = datedict[fname.stem[:10]][1] + large_error
                
                # when all hourly velocities of a day are included, the error is
                # set to 1.5 if more large errors than small ones were found
                # like this, only days with many decorrelations get a decorrelation
                if datedict[fname.stem[:10]][0] > 12:
                    velarray[datedict[fname.stem[:10]][1]>10,4] = 1.5
                    # else the error is 0.555 to show that there is no filtering
                    velarray[datedict[fname.stem[:10]][1]<10,4] = 0.555
                
                # # set the max error to the error of the considered measurement
                # # if the error is larger. If there was a decorrelation in that 
                # # day, it should be considered in the daily results
                # velarray[velarray[:,4]<res[:,4],4] = res[velarray[:,4]<res[:,4],4]
                
                velo = pd.DataFrame({'start u [px]':velarray[:,0], 'start v [px]':velarray[:,1],
                                     'daily average du [px/h]':velarray[:,2], 'daily average dv [px/h]':velarray[:,3],
                                     'max error':velarray[:,4]})
                # save the resulting pandas DataFrame to csv
                velo.to_csv(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')),index=False)
                
                # np.savetxt(str(self._outpath.joinpath('TS-'+fname.stem[:10]+'.csv')), velarray, delimiter=",")
                
                datedict[fname.stem[:10]][0] = datedict[fname.stem[:10]][0] + 1
       
        if self._ts_filt:
            
            print('--- Use Time Series Filter')
            
            # get all paths to the velocity per hour time series results
            input_list = [p for p in self._outpath.iterdir() if p.suffix.lower() == '.csv' and
                         str(p.stem).startswith('TS')]
            input_list.sort()
            
            # more than 6 days needed since else too few values for comparison
            if len(input_list) > 6:
            
                for ii in range(len(input_list)):
                    if ii == 0:
                        
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts1 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii+2]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii+3]), delimiter=',')[1:,:] 
                        ts4 = np.genfromtxt(str(input_list[ii+4]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii+5]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii+6]), delimiter=',')[1:,:] 
                    
                    elif ii == 1:
                        
                        ts1 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii+2]), delimiter=',')[1:,:] 
                        ts4 = np.genfromtxt(str(input_list[ii+3]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii+5]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii+4]), delimiter=',')[1:,:] 
    
                    elif ii == 2:
                        
                        ts1 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii+2]), delimiter=',')[1:,:] 
                        ts4 = np.genfromtxt(str(input_list[ii+3]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii-2]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii+4]), delimiter=',')[1:,:]                     
                
                    elif ii == len(input_list)-1:
                        
                        ts3 = np.genfromtxt(str(input_list[ii-3]), delimiter=',')[1:,:] 
                        ts1 = np.genfromtxt(str(input_list[ii-2]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts4 = np.genfromtxt(str(input_list[ii-6]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii-4]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii-5]), delimiter=',')[1:,:] 
                        
                    elif ii == len(input_list)-2:
                        
                        ts4 = np.genfromtxt(str(input_list[ii-4]), delimiter=',')[1:,:] 
                        ts1 = np.genfromtxt(str(input_list[ii-3]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii-2]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii-5]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        
                    elif ii == len(input_list)-3:
                        
                        ts4 = np.genfromtxt(str(input_list[ii-4]), delimiter=',')[1:,:] 
                        ts1 = np.genfromtxt(str(input_list[ii-3]), delimiter=',')[1:,:] 
                        ts2 = np.genfromtxt(str(input_list[ii-2]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii+2]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        
                    else:
                        
                        ts2 = np.genfromtxt(str(input_list[ii-2]), delimiter=',')[1:,:] 
                        ts1 = np.genfromtxt(str(input_list[ii-1]), delimiter=',')[1:,:] 
                        ts0 = np.genfromtxt(str(input_list[ii]), delimiter=',')[1:,:] 
                        ts3 = np.genfromtxt(str(input_list[ii+1]), delimiter=',')[1:,:] 
                        ts4 = np.genfromtxt(str(input_list[ii+2]), delimiter=',')[1:,:] 
                        ts5 = np.genfromtxt(str(input_list[ii+3]), delimiter=',')[1:,:] 
                        ts6 = np.genfromtxt(str(input_list[ii-3]), delimiter=',')[1:,:] 
                        
                    if np.all(np.isnan(ts4[:,2])):
                        ts4[:,2:4] = 0
                    if np.all(np.isnan(ts3[:,2])):
                        ts3[:,2:4] = 0
                    if np.all(np.isnan(ts5[:,2])):
                        ts5[:,2:4] = 0
                    if np.all(np.isnan(ts6[:,2])):
                        ts6[:,2:4] = 0
                    if np.all(np.isnan(ts2[:,2])):
                        ts2[:,2:4] = 0
                    if np.all(np.isnan(ts1[:,2])):
                        ts1[:,2:4] = 0
                    if np.all(np.isnan(ts0[:,2])):
                        continue
                    
                    
                    mag0 = (ts0[:,2]**2+ts0[:,3]**2)**0.5
                    mag1 = (ts1[:,2]**2+ts1[:,3]**2)**0.5
                    mag2 = (ts2[:,2]**2+ts2[:,3]**2)**0.5
                    mag3 = (ts3[:,2]**2+ts3[:,3]**2)**0.5
                    mag4 = (ts4[:,2]**2+ts4[:,3]**2)**0.5
                    mag5 = (ts5[:,2]**2+ts5[:,3]**2)**0.5
                    mag6 = (ts6[:,2]**2+ts6[:,3]**2)**0.5
                    
                    mag_m = (mag1+mag2+mag3+mag4+mag0+mag5+mag6)/7
                    mag_thr = mag0/2
                    mag_out = np.zeros(mag_thr.shape,dtype=bool)
                    mag_out[mag_thr>mag_m] = True
                    
                    y_m = (ts0[:,3]+ts1[:,3]+ts2[:,3]+ts3[:,3]+ts4[:,3]+ts5[:,3]+ts6[:,3])/7
                    y_thr = ts0[:,3]/2
                    y_out = np.zeros(y_thr.shape,dtype=bool)
                    y_out[(y_thr<0)&(y_m>=y_thr)] = True
                    y_out[(y_thr>0)&(y_m<=y_thr)] = True
                    
                    ts0[mag_out,2:4] = np.nan
                    ts0[y_out,2:4] = np.nan
                    
                    ts_filt = pd.DataFrame({'start u [px]':ts0[:,0], 'start v [px]':ts0[:,1],
                                     'du [px/h]':ts0[:,2], 'dv [px/h]':ts0[:,3],
                                     'min error':ts0[:,4]})
                    # save the resulting pandas DataFrame to csv
                    ts_filt.to_csv(str(self._outpath.joinpath(input_list[ii].stem)),index=False)
            
            
        print('--- Cumulate displacements per day')
        
        # list of the time series filenames of velocity per hour averaged for each day
        in_list = [p for p in self._outpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).startswith('TS-')]
        in_list.sort()
        
        cum_disp = np.zeros(velarray.shape)
        # cum_disp = np.zeros((4941,5))
        # cum_disp[:,:2] = velarray[:,:2]
        counter = 0
        for fname in in_list:
            

            # add CUM to the output name of the cumulated results 
            exname = fname.parent.joinpath('CUM-'+fname.stem+'.csv')
            
            # read velocity per hour averaged for each day
            try:
                velarray = np.genfromtxt(str(fname), delimiter=',')[1:,:]
            except:
                velarray = np.genfromtxt(str(fname), delimiter=';')[1:,:]
            
            if counter == 0:
                cum_disp[:,:2] = velarray[:,:2]
                
            counter += 1
            
            # add 24 times the average velocity per hour to get the displacement of the day
            cum_disp[~np.isnan(velarray[:,2]),2:4] = cum_disp[~np.isnan(velarray[:,2]),2:4] + velarray[~np.isnan(velarray[:,2]),2:4] * 24
            
            # alternatively
            # cum_disp[2:] = cum_disp[2:] + vel[2:]*datedict[fname.stem.split('-')[-1]]
            
            # change the error to the value to the one of which the displacement
            # was just added
            cum_disp[~np.isnan(velarray[:,4]),4] = velarray[~np.isnan(velarray[:,4]),4]
            cum_disp[np.isnan(velarray[:,4]),4] = 0.999
            
            
            velo = pd.DataFrame({'start u [px]':cum_disp[:,0], 'start v [px]':cum_disp[:,1],
                                 'cumulative displacement du [px]':cum_disp[:,2],
                                 'cumulative displacement dv [px]':cum_disp[:,3],
                                 'max error': cum_disp[:,4]})
            # save the resulting pandas DataFrame to csv
            velo.to_csv(exname,index=False)
                
            # np.savetxt(exname, cum_disp, delimiter=",")
            
            
        
    def extract_pointTS(self,pointname,pointx,pointy):
        """
        extracts the information of a point from cumulated displacements and
        saves them

        Parameters
        ----------
        pointname : string
            name of the point that should be extract (output file name).
        pointx : integer
            x location in pixel units of the point that should be extracted.
        pointy : integer
            y location in pixel units of the point that should be extracted..

        Returns
        -------
        None.

        """
        cum_list = [p for p in self._outpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).startswith('3D-CUM')]
        # create an array to temporarily save the point results
        tempsave = np.empty((len(cum_list),11))
        tempdate = list()
        
        # loop through all 3D results an get the results at the point location
        for ii,fname in enumerate(cum_list):
            try:
                res = np.genfromtxt(str(fname), delimiter=',')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            except:
                res = np.genfromtxt(str(fname), delimiter=';')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            results = res[(res[:,0]==pointx)&(res[:,1]==pointy),:]
            
            # save du, dv, dtot2d and X1,X2,X3 and dx,dy,dz,dtot3d and error in tempsave array
            tempsave[ii,0:3] = results[:,2:5]
            tempsave[ii,3:6] = results[:,5:8]
            tempsave[ii,6:-2] = results[:,-5:-2]
            tempsave[ii,-2] = (results[:,-5]**2+results[:,-4]**2+results[:,-3]**2)**0.5
            tempsave[ii,-1] = results[:,-1]
            
            # get the dates of the 3D results name for indexing
            tempdate.append(Path(fname).stem.split('-')[-3]+'-'+
                            Path(fname).stem.split('-')[-2]+'-'+
                            Path(fname).stem.split('-')[-1])
            
        # save temporary results in a pandas dataframe and save it as csv
        pointDF = pd.DataFrame({'dates':tempdate, 'du ':tempsave[:,0], 'dv ':tempsave[:,1],
                             'd2-tot':tempsave[:,2], 'X1':tempsave[:,3],'Y1':tempsave[:,4],
                             'Z1':tempsave[:,5],'dX':tempsave[:,6],'dY':tempsave[:,7],
                             'dZ':tempsave[:,8], 'd3-tot':tempsave[:,9], 'max error':tempsave[:,10]})
        
        # define exname into output path with pointname
        exname = self._outpath.joinpath('3D-extracted-CUM-' + pointname + '.csv')
        
        pointDF.to_csv(str(exname),index=False)
        
        
    def createPointPlot(self):
        """
        Creates a plot for every point extracted from the cumulative displacements

        Returns
        -------
        None.

        """
        
        point_list = [p for p in self._outpath.iterdir() if p.suffix.lower() == '.csv' and
                     str(p.stem).startswith('3D-extracted')]
        
        for fname in point_list:
        
            try:
                # columns: start u, start v, du, dv, d2d, dX, dY, dZ, dTot
                res = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
                resdates = np.genfromtxt(str(fname), delimiter=',',dtype=str)[1:,:]
            except:
                res = np.genfromtxt(str(fname), delimiter=';')[1:,:]
                resdates = np.genfromtxt(str(fname), delimiter=';',dtype=str)[1:,:]
            
            datelist = list()
            for dats in resdates[:,0]:
                try:
                    dati = dats.split('.')
                    datelist.append(date(int(dati[2]),int(dati[1]),int(dati[0])))
                except:
                    dati = dats.split('-')
                    datelist.append(date(int(dati[0]),int(dati[1]),int(dati[2])))
                
            # rockfalls = np.zeros((res1d.shape[0],))*np.nan
            # rockfalls[res1d[:,-1]>=1] = 80
            
            rockfalls = resdates[res[:,-1]>=1,0]
            rockfall = list()
            for dats in rockfalls:
                try:
                    dati = dats.split('.')
                    rockfall.append(date(int(dati[2]),int(dati[1]),int(dati[0])))
                except:
                    dati = dats.split('-')
                    rockfall.append(date(int(dati[0]),int(dati[1]),int(dati[2])))
                
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        
            
            # rockf = ax.plot(datelist, rockfalls, color='black', marker='vline',
            #                 markersize=100, alpha = 0.5)
            
            
            
            for dats in rockfall:
                ax.axvline(x=dats, color='black')
            magn = ax.plot(datelist,res[:,-2],label='3D Magnitude')
            
            ax.legend()
            ax.grid()
            
            ax.set_xlabel('Date')
            ax.set_ylabel('3D Magnitude [m]')
            
            ax.set_ylim(0, np.nanmax(res[:,-2]+5))
            
            fig.autofmt_xdate(rotation=90)
            
            plt.gcf().set_dpi(500)
            plt.tight_layout()
            
            plt.savefig(self._outpath.joinpath(fname.stem+'.jpg'))
            plt.show()
    

    def createGIF(self,mask_path):
        
        # get a discrete colorbap with pylab.cm to create a good gif
        cmap = cm.get_cmap('jet',50)
        
        greymap = cm.get_cmap('Greys_r',50)

        
        # limits of the visualizations, 0 for de-cumulated, 1 for cumulated
        PLTMIN0 = 0.1
        PLTMAX0 = 1
        PLTMIN1 = 1
        PLTMAX1 = 10
        # PLTMIN0 = 0.1
        # PLTMAX0 = 2.5
        # PLTMIN1 = 5
        # PLTMAX1 = 45
        
        # settings for the colorbar location
        aspect = 20
        pad_fraction = 0.5
        
        # # get the path to the mask
        # mask = [p for p in mask_path.iterdir() if p.suffix.lower() == '.csv' and 
        #         'mask' in str(p.stem)]
        
        # # read the mask
        # mask = np.genfromtxt(str(mask[0]), delimiter=',')[1:,:] 
        
        # path to the output folder (GIF folder within output folder)
        out_path = self._outpath.parent.joinpath("GIF")
        out_path.mkdir(parents=True, exist_ok=True)
        
        # get all paths to the 3D cumulated displacements in the time-series 
        # folder within the output folder
        disp_list = [p for p in self._inpath.joinpath('time-series').iterdir() if p.suffix.lower() == '.csv' and
                     str(p.stem).startswith('3D-CUM')]
        
        # get all the background images in the image folder within the output folder
        img_list = [p for p in self._inpath.joinpath('images').iterdir() if p.suffix.lower() == '.jpg' and 
                    str(p.stem).startswith('20')]
        
        # get the first background image, which will be used as long as there 
        # is not other background corresponding to the date found
        img0 = cv2.imread(str(img_list[0]), cv2.IMREAD_GRAYSCALE)
        
        # "empty" array which will be used to de-cumulate the displacements
        uncum_disp = np.array([[100000],[10]])
        
        # create one plot per day with the cumulated or decumulated displacememnts
        if self.SINGLEPLOT:
            for iii,fname in enumerate(disp_list):
                
                # the title tit, namely the date of the image/displacement
                tit = fname.stem[3:]
                
                try:
                    ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
                except:
                    ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                    
                #_________________________________________________________________________
                #  # uncomment this for de-cumulation of displacement results
                # if uncum_disp[0,0] == 100000:
                #     uncum_disp = np.zeros(ts00.shape)
            
                # ts0 = copy.deepcopy(ts00)
                # ts0[:,2:5] = ts00[:,2:5] - uncum_disp[:,2:5]
                # ts0[:,11:15] = ts00[:,11:15] - uncum_disp[:,11:15]
                
                # # change the limits since its now cumulated
                # PLTMIN0 = PLTMIN1
                # PLTMAX0 = PLTMAX1
            
                # uncum_disp = copy.deepcopy(ts00)
                #_________________________________________________________________________
                
                # comment this when the de-cumulated displacements should be shown
                ts0 = ts00

                # # in the first iteration, the mask should be reshaped to 
                # # the shape and grid size of the results
                # if mask.shape == (1943,2592):
                #     bool_disp = np.zeros((ts0.shape[0],),dtype=bool)
                #     for ii in range( ts0.shape[0]):
                #         if mask[int(ts0[ii,1]),int(ts0[ii,0])] == 1: 
                #             bool_disp[ii] = True
                #     mask = bool_disp
                
                # filter the time series results with the mask, the error, and
                # with the magnitude in 3D
                # disp = ts0[mask&(ts0[:,-1]<=1)&(ts0[:,-2]<=PLTMAX0)&(ts0[:,-2]>PLTMIN0),:]
                # filt_disp = ts0[mask&(ts0[:,-1]>1)&(ts0[:,-2]<=PLTMAX0)&(ts0[:,-2]>PLTMIN0),:]
                disp = ts0[(ts0[:,-1]<=1)&(ts0[:,-2]<=PLTMAX0)&(ts0[:,-2]>PLTMIN0),:]
                filt_disp = ts0[(ts0[:,-1]>1)&(ts0[:,-2]<=PLTMAX0)&(ts0[:,-2]>PLTMIN0),:]
                
                # update the background image if there is one in the image folder
                for imgname in img_list:
                    if fname.stem.split('TS')[1][1:] in str(imgname):
                        img0 = cv2.imread(str(imgname), cv2.IMREAD_GRAYSCALE)
            
                # create a figure with a single plot and axis
                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7))
                
                # display the background image
                im0 = ax.imshow(img0,cmap=greymap,alpha=0.3)
                
                # if wanted: plot arrows
                # sca0 = ax[0].quiver(ts0[:,1],ts0[:,0],ts0[:,3],ts0[:,2])
                # sca0 = ax.quiver(disp[:,0],disp[:,1],disp[:,2],-disp[:,3],disp[:,-2],
                #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
                
                # plot the points colored with the 3D magnitude
                sca0 = ax.scatter(disp[:,0],disp[:,1],c=disp[:,-2],cmap='jet', vmin=0, vmax=20,s=3)
                # plot the filtered results as outliers or decorrelations
                sca1 = ax.plot(filt_disp[:,0],filt_disp[:,1],'.',ms=5,c='magenta')
                
                # plot the date on the image
                t = plt.text(0.01, 0.95, fname.stem.split('TS')[1][1:], size=13, color='black', transform=ax.transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
                # change the limits of the axes and the direction of y-axis
                plt.xlim([0, 2570])
                plt.ylim([0, 1944])
                ax.invert_yaxis()
                
                # add a colorbar and define its position
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1./aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                plt.colorbar(sca0, cax=cax, label='Deformation (pixels)')
                
                # change the resolution of the plot
                plt.gcf().set_dpi(300)
                
                # save the images
                if iii < 10:
                    plt.savefig(str(out_path.joinpath('single-gif00' + str(iii) + '.jpg')),bbox_inches='tight')
                elif iii < 100:
                    plt.savefig(str(out_path.joinpath('single-gif0' + str(iii) + '.jpg')),bbox_inches='tight')
                else:
                    plt.savefig(str(out_path.joinpath('single-gif' + str(iii) + '.jpg')),bbox_inches='tight')
                plt.show()
            
        if self.MULTIPLOT:
            for iii,fname in enumerate(disp_list):
                
                PLTMIN0 = self.scale_3d[0]
                PLTMAX0 = self.scale_3d[1]
                PLTMIN1 = self.scale_3d[2]
                PLTMAX1 = self.scale_3d[3]
                
                # the title tit, namely the date of the image/displacement
                tit = fname.stem[3:]
                
                try:
                    ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
                except:
                    ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                    
                # set the matrix for de-cumulation to zeros in the first iteration
                if uncum_disp[0,0] == 100000:
                    uncum_disp = np.zeros(ts00.shape)
            
                # decumulate the displacement measurements with the values of the
                # last iteration
                ts1 = copy.deepcopy(ts00)
                ts1[:,2:5] = ts00[:,2:5] - uncum_disp[:,2:5]
                ts1[:,11:15] = ts00[:,11:15] - uncum_disp[:,11:15]
            
                # # redefine variable for next iteration
                # uncum_disp = copy.deepcopy(ts00)
                
                # cumulated results for the right plot
                ts0 = ts00
                
        
                
                # if mask.shape == (1943,2592):
                #     bool_disp = np.zeros((ts0.shape[0],),dtype=bool)
                #     for ii in range( ts0.shape[0]):
                #         if mask[int(ts0[ii,1]),int(ts0[ii,0])] == 1: 
                #             bool_disp[ii] = True
                #     mask = bool_disp
                
                # disp0 = ts0[mask&(ts0[:,-1]<=1)&(d20>0.5)&(ts0[:,-2]<=PLTMAX1)&(ts0[:,-2]>2),:]
                # filt_disp0 = ts0[mask&(ts0[:,-1]>1)&(ts0[:,-2]<=20)&(ts0[:,-2]>-0.001),:]
                
                # disp1 = ts1[mask&(ts1[:,-1]<=1)&(d21>0.5)&(ts1[:,-2]<=20)&(ts1[:,-2]>0.5),:]
                # filt_disp1 = ts1[mask&(ts1[:,-1]>1)&(ts1[:,-2]<=20)&(ts1[:,-2]>-0.001),:]
                
                disp0 = ts0[(ts0[:,-1]<=1)&(ts0[:,-2]<=PLTMAX1)&(ts0[:,-2]>PLTMIN1),:]
                filt_disp = ts0[(ts0[:,-1]>1),:]
                
                disp1 = ts1[(ts1[:,-1]<=1)&(ts1[:,-2]<=PLTMAX0)&(ts1[:,-2]>PLTMIN0),:]
                # filt_disp1 = ts1[(ts1[:,-1]>1)&(ts1[:,-2]<=20)&(ts1[:,-2]>-0.001),:]
                
                
                # ts10max = ts00[(ts00[:,-1]>PLTMAX),:]
                # ts00[(ts00[:,-1]>PLTMAX),:] = np.nan
                # ts00[(ts00[:,-1]<PLTMIN),:] = np.nan
                
                for imgname in img_list:
                    if fname.stem.split('TS')[1][1:] in str(imgname):
                        img0 = cv2.imread(str(imgname), cv2.IMREAD_GRAYSCALE)
            
                    
                fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10.5,3))
                
                im0 = ax[0].imshow(img0,cmap='gray',alpha=0.3)
                
                # sca0 = ax.quiver(disp0[:,0],disp0[:,1],disp0[:,2],-disp0[:,3],disp0[:,-2],
                #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
                sca00 = ax[0].scatter(disp0[:,0],disp0[:,1],c=disp0[:,-2],cmap=cmap, vmin=PLTMIN1, vmax=PLTMAX1,s=3)
                sca10 = ax[0].plot(filt_disp[:,0],filt_disp[:,1],'.',ms=2,c='magenta')
                
                t = plt.text(0.01, 0.95, fname.stem.split('TS')[1][1:], size=9, color='black', transform=ax[0].transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
                im0 = ax[1].imshow(img0,cmap='gray',alpha=0.3)
                
                # sca0 = ax.quiver(disp1[:,0],disp1[:,1],disp1[:,2],-disp1[:,3],disp1[:,-2],
                #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
                sca01 = ax[1].scatter(disp1[:,0],disp1[:,1],c=disp1[:,-2],cmap=cmap, vmin=PLTMIN0, vmax=PLTMAX0,s=3)
                sca11 = ax[1].plot(filt_disp[:,0],filt_disp[:,1],'.',ms=2,c='magenta')
                
                # plt.xlim([0, 2570])
                # plt.ylim([0, 1944])
                # ax.invert_yaxis()
                
                divider = make_axes_locatable(ax[0])
                width = axes_size.AxesY(ax[0], aspect=1./aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                plt.colorbar(sca00, cax=cax, label='Cumulative Displacement (m)')
                
                divider = make_axes_locatable(ax[1])
                width = axes_size.AxesY(ax[1], aspect=1./aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                plt.colorbar(sca01, cax=cax, label='Velocity (m/d)')
                
                
                # plt.gca().invert_xaxis()
                
                plt.gcf().set_dpi(300)
                
                # sca0 = ax[0].plot(ts0[:,1],ts0[:,0])
                if iii < 10:
                    plt.savefig(str(out_path.joinpath('3D-multi-gif00' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                elif iii < 100:
                    plt.savefig(str(out_path.joinpath('3D-multi-gif0' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                else:
                    plt.savefig(str(out_path.joinpath('3D-multi-gif' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                plt.show()
                
                ################################################################
                # now for 2D
                ################################################################
                
                PLTMIN0 = self.scale_2d[0]
                PLTMAX0 = self.scale_2d[1]
                PLTMIN1 = self.scale_2d[2]
                PLTMAX1 = self.scale_2d[3]

                try:
                    ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
                except:
                    ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                    
                # set the matrix for de-cumulation to zeros in the first iteration
                if uncum_disp[0,0] == 100000:
                    uncum_disp = np.zeros(ts00.shape)
            
                # decumulate the displacement measurements with the values of the
                # last iteration
                ts1 = copy.deepcopy(ts00)
                ts1[:,2:5] = ts00[:,2:5] - uncum_disp[:,2:5]
                ts1[:,11:15] = ts00[:,11:15] - uncum_disp[:,11:15]
            
                # redefine variable for next iteration
                uncum_disp = copy.deepcopy(ts00)
                
                # cumulated results for the right plot
                ts0 = ts00
                
                disp0 = ts0[(ts0[:,-1]<=1)&(ts0[:,4]<=PLTMAX1)&(ts0[:,4]>PLTMIN1),:]
                filt_disp = ts0[(ts0[:,-1]>1),:]
                
                disp1 = ts1[(ts1[:,-1]<=1)&(ts1[:,4]<=PLTMAX0)&(ts1[:,4]>PLTMIN0),:]
                # filt_disp1 = ts1[(ts1[:,-1]>1)&(ts1[:,4]<=20)&(ts1[:,4]>-0.001),:]
                
                # ts10max = ts00[(ts00[:,-1]>PLTMAX),:]
                # ts00[(ts00[:,-1]>PLTMAX),:] = np.nan
                # ts00[(ts00[:,-1]<PLTMIN),:] = np.nan
                
                for imgname in img_list:
                    if fname.stem.split('TS')[1][1:] in str(imgname):
                        img0 = cv2.imread(str(imgname), cv2.IMREAD_GRAYSCALE)
            
                    
                fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10.5,3))
                
                im0 = ax[0].imshow(img0,cmap=greymap,alpha=0.4)
                
                # sca0 = ax.quiver(disp0[:,0],disp0[:,1],disp0[:,2],-disp0[:,3],disp0[:,-2],
                #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
                sca00 = ax[0].scatter(disp0[:,0],disp0[:,1],c=disp0[:,4],cmap=cmap, vmin=PLTMIN1, vmax=PLTMAX1,s=3,alpha=0.8)
                sca10 = ax[0].plot(filt_disp[:,0],filt_disp[:,1],'.',ms=2,c='magenta')
                
                # optionally plot some point locations
                
                # ax[0].plot(1312,288,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(768,736,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(1120,736,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(1152,960,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(864,1088,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(544,1152,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(1088,1312,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(608,1472,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(1024,1568,'o', mfc='none', ms=4, c= 'r')
                # ax[0].plot(1344,1664,'o', mfc='none', ms=4, c= 'r')


                
                t = plt.text(0.01, 0.95, fname.stem.split('TS')[1][1:], size=9, color='black', transform=ax[0].transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
                
                im0 = ax[1].imshow(img0,cmap=greymap,alpha=0.4)
                
                # sca0 = ax.quiver(disp1[:,0],disp1[:,1],disp1[:,2],-disp1[:,3],disp1[:,-2],
                #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
                sca01 = ax[1].scatter(disp1[:,0],disp1[:,1],c=disp1[:,4],cmap=cmap, vmin=PLTMIN0, vmax=PLTMAX0,s=3,alpha=0.8)
                sca11 = ax[1].plot(filt_disp[:,0],filt_disp[:,1],'.',ms=2,c='magenta')
                
                # optionally plot some point locations
                
                # ax[1].plot(1312,288,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(768,736,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(1120,736,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(1152,960,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(864,1088,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(544,1152,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(1088,1312,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(608,1472,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(1024,1568,'o', mfc='none', ms=4, c= 'r')
                # ax[1].plot(1344,1664,'o', mfc='none', ms=4, c= 'r')
                
                # plt.xlim([0, 2570])
                # plt.ylim([0, 1944])
                # ax.invert_yaxis()
                
                divider = make_axes_locatable(ax[0])
                width = axes_size.AxesY(ax[0], aspect=1./aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                plt.colorbar(sca00, cax=cax, label='Cumulative Displacement (px)')
                
                divider = make_axes_locatable(ax[1])
                width = axes_size.AxesY(ax[1], aspect=1./aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                plt.colorbar(sca01, cax=cax, label='Velocity (px/d)')
                
                
                # plt.gca().invert_xaxis()
                
                plt.gcf().set_dpi(300)
                
                # sca0 = ax[0].plot(ts0[:,1],ts0[:,0])
                if iii < 10:
                    plt.savefig(str(out_path.joinpath('2D-multi-gif00' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                elif iii < 100:
                    plt.savefig(str(out_path.joinpath('2D-multi-gif0' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                else:
                    plt.savefig(str(out_path.joinpath('2D-multi-gif' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
                plt.show()
            
        if self.SINGLEPLOT:
            import imageio
            filenames = [p for p in out_path.iterdir() if p.suffix.lower() == '.jpg' and str(p.stem).startswith('single')]
            # filenames.sort()
            with imageio.get_writer(str(out_path.joinpath('singleplot.gif')), mode='I',duration=0.5) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    
        if self.MULTIPLOT:
            
            filenames = [p for p in out_path.iterdir() if p.suffix.lower() == '.jpg' and str(p.stem).startswith('3D-multi')]
            frame = cv2.imread(str(filenames[0]))
            height, width, layers = frame.shape
            video = cv2.VideoWriter(str(out_path.joinpath('3D-multiplot.mp4')), 0, 1, (width,height))
            for fname in filenames:
                
                video.write(cv2.imread(str(fname)))
                
            cv2.destroyAllWindows()
            video.release()
            # filenames.sort()
            import imageio
            with imageio.get_writer(str(out_path.joinpath('3D-multiplot.gif')), mode='I',duration=0.5) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    
                    
            filenames = [p for p in out_path.iterdir() if p.suffix.lower() == '.jpg' and str(p.stem).startswith('2D-multi')]
            frame = cv2.imread(str(filenames[0]))
            height, width, layers = frame.shape
            video = cv2.VideoWriter(str(out_path.joinpath('2D-multiplot.mp4')), 0, 1, (width,height))
            for fname in filenames:
                
                video.write(cv2.imread(str(fname)))
                
            cv2.destroyAllWindows()
            video.release()
            # filenames.sort()
            with imageio.get_writer(str(out_path.joinpath('2D-multiplot.gif')), mode='I',duration=0.5) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

    def extract_decorr(self):
        
        disp_list = [p for p in self._inpath.joinpath('time-series').iterdir() if p.suffix.lower() == '.csv' and
                     str(p.stem).startswith('3D-CUM')]
        
        decor = np.ones((len(disp_list),4))
        
        for ii,fname in enumerate(disp_list):
            try:
                ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
            except:
                ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                
            dates = fname.stem.split('-')
            date = dates[-3]+dates[-2]+dates[-1]
            filt_disp = ts00[(ts00[:,-1]>1),:]
            num_decor = filt_disp.shape[0]
            
            x_decorr = np.nanmean(filt_disp[:,0])
            y_decorr = np.nanmean(filt_disp[:,1])
            
            # add to the decorrelation array: date, number of decorrelated patches,
            # mean x and y (pixel) location of decorrelated patches
            decor[ii,0] = date
            decor[ii,1] = num_decor
            decor[ii,2] = x_decorr
            decor[ii,3] = y_decorr

        np.savetxt(self._inpath.joinpath('000_extracted_decorrelation.csv'),decor,delimiter=',')
    
    def inclinationGIF(self):
        # get a discrete colorbap with pylab.cm to create a good gif
        cmap = cm.get_cmap('brg',50)
        greymap = cm.get_cmap('Greys_r',50)
        
        # limits of the visualizations (min and max inclinations)
        PLTMIN0 = -3
        PLTMAX0 = 1

        
        # settings for the colorbar location
        aspect = 20
        pad_fraction = 0.5
        
        # path to the output folder (GIF folder within output folder)
        out_path = self._outpath.parent.joinpath("inclination")
        out_path.mkdir(parents=True, exist_ok=True)
        
        # get all paths to the 3D cumulated displacements in the time-series 
        # folder within the output folder
        disp_list = [p for p in self._inpath.joinpath('time-series').iterdir() if p.suffix.lower() == '.csv' and
                     str(p.stem).startswith('3D-CUM')]
        
        # get all the background images in the image folder within the output folder
        img_list = [p for p in self._inpath.joinpath('images').iterdir() if p.suffix.lower() == '.jpg' and 
                    str(p.stem).startswith('20')]
        
        # get the first background image, which will be used as long as there 
        # is not other background corresponding to the date found
        img0 = cv2.imread(str(img_list[0]), cv2.IMREAD_GRAYSCALE)
        
        # "empty" array which will be used to de-cumulate the displacements
        uncum_disp = np.array([[100000],[10]])
        inclination = np.array([[100000],[10]])
        
        for iii,fname in enumerate(disp_list):
            
            try:
                ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
            except:
                ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                

            if uncum_disp[0,0] == 100000:
                uncum_disp = np.zeros(ts00.shape)
                
            if inclination[0,0] == 100000:
                inclination = np.zeros((ts00.shape[0],4))
                inclination[:,:2] = ts00[:,:2]
        
            ts0 = copy.deepcopy(ts00)
            ts0[:,2:5] = ts00[:,2:5] - uncum_disp[:,2:5]
            ts0[:,11:15] = ts00[:,11:15] - uncum_disp[:,11:15]
        
            uncum_disp = copy.deepcopy(ts00)

            disp = ts0
            
            # calculate the inclination of the 3D world coordinate vector
            incli = disp[:,-3]/((disp[:,-5]**2+disp[:,-4]**2)**0.5)
            
            # save the inclination to a csv to extract the evolution of 
            # inclincation with time at point locations
            savincli = np.zeros((ts00.shape[0],3))
            savincli[:,:2] = ts00[:,:2]
            savincli[:,2] = incli
            np.savetxt(str(out_path.joinpath('incli-'+fname.stem.split('TS')[1][1:]+'.csv')),savincli,delimiter=',')
            
            
            # calculate the mean inclination
            inclination[~np.isnan(incli),2] = (inclination[~np.isnan(incli),2]*iii+incli[~np.isnan(incli)])/(iii+1) 
            
            # calculate the mean 2D magnitude for plotting of mean inclination
            inclination[~np.isnan(disp[:,4]),3] = (inclination[~np.isnan(disp[:,4]),3]*iii+disp[~np.isnan(disp[:,4]),4])/(iii+1)
            
            # filter the time series results with the mask, the error, and
            # with the magnitude in 3D
            # disp = ts0[(ts0[:,-1]<=1)&(ts0[:,4]>0.1)&(ts0[:,4]<5),:]
            # filt_disp = ts0[(ts0[:,-1]>1),:]

            # cut small and large inclinations for visualization
            # as well, cut the large and small 2d magnitudes
            disp = disp[(incli>-10)&(incli<10)&(ts0[:,4]>0.07)&(ts0[:,4]<5),:]
            incli = incli[(incli>-10)&(incli<10)&(ts0[:,4]>0.07)&(ts0[:,4]<5)]
            
            # update the background image if there is one in the image folder
            for imgname in img_list:
                if fname.stem.split('TS')[1][1:] in str(imgname):
                    img0 = cv2.imread(str(imgname), cv2.IMREAD_GRAYSCALE)
        
            # create a figure with a single plot and axis
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5.5))
            
            # display the background image
            im0 = ax.imshow(img0,cmap=greymap,alpha=0.3)
            
            # if wanted: plot arrows
            # sca0 = ax[0].quiver(ts0[:,1],ts0[:,0],ts0[:,3],ts0[:,2])
            # sca0 = ax.quiver(disp[:,0],disp[:,1],disp[:,2],-disp[:,3],disp[:,-2],
            #            scale=None,width=0.01,units='inches',cmap='jet',clim=(0,30))
            
            # plot the points colored with the 3D magnitude
            sca0 = ax.scatter(disp[:,0],disp[:,1],c=incli,cmap=cmap, vmin=PLTMIN0, vmax=PLTMAX0,s=3)

            
            # plot the date on the image
            t = plt.text(0.01, 0.95, fname.stem.split('TS')[1][1:], size=13, color='black', transform=ax.transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))
            
            # change the limits of the axes and the direction of y-axis
            plt.xlim([0, 2570])
            plt.ylim([0, 1944])
            ax.invert_yaxis()
            
            # add a colorbar and define its position
            divider = make_axes_locatable(ax)
            width = axes_size.AxesY(ax, aspect=1./aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.colorbar(sca0, cax=cax, label='Inclination []')
            
            # change the resolution of the plot
            plt.gcf().set_dpi(300)
            
            # save the images
            if iii < 10:
                plt.savefig(str(out_path.joinpath('inclination00' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
            elif iii < 100:
                plt.savefig(str(out_path.joinpath('inclination0' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
            else:
                plt.savefig(str(out_path.joinpath('inclination' + str(iii) + '.jpg')),bbox_inches='tight',dpi=300)
            plt.show()
            

        # save the inclination gif images as a gif and as a mp4
        filenames = [p for p in out_path.iterdir() if p.suffix.lower() == '.jpg' and str(p.stem).startswith('incli')]
        frame = cv2.imread(str(filenames[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(str(out_path.joinpath('inclination.mp4')), 0, 1, (width,height))
        for fname in filenames:
            
            video.write(cv2.imread(str(fname)))
            
        cv2.destroyAllWindows()
        video.release()
        # filenames.sort()
        with imageio.get_writer(str(out_path.joinpath('inclination.gif')), mode='I',duration=0.5) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        # calculate the standard deviation of inclination
        var = np.array([[100000],[10]])
        
        for iii,fname in enumerate(disp_list):

            
            try:
                ts00 = np.genfromtxt(str(fname), delimiter=',')[1:,:] 
            except:
                ts00 = np.genfromtxt(str(fname), delimiter=';')[1:,:] 
                
            if var[0,0] == 100000:
                var = np.zeros((ts00.shape[0],3))
                var[:,:2] = ts00[:,:2]
        
            ts0 = copy.deepcopy(ts00)
            ts0[:,2:5] = ts00[:,2:5] - uncum_disp[:,2:5]
            ts0[:,11:15] = ts00[:,11:15] - uncum_disp[:,11:15]

            disp = ts0
            
            # calculate the inclination of the 3D world coordinate vector
            incli = disp[:,-3]/((disp[:,-5]**2+disp[:,-4]**2)**0.5)
            # breakpoint()
            # calculate the deviation of a specific inclination and thereafter
            # include it in the calculation of the variance
            dev = (inclination[:,2]-incli)**2
            var[~np.isnan(dev),2] = (var[~np.isnan(dev),2]*iii+dev[~np.isnan(dev)])/(iii+1) 
        
        # get the standard deviation from the variance
        stddev = copy.deepcopy(var)
        stddev[:,2] = (var[:,2])**0.5
        
        # get rid of inclinations with low std dev & 2D magnitude
        stddev = stddev[inclination[:,-1]>0.03,:]
        stddev = stddev[stddev[:,2]>0,:]
        stddev = stddev[stddev[:,2]<10,:]
        
        # get rid of inclinations with low mean 3D magnitude
        inclination = inclination[inclination[:,-1]>0.03,:]
        inclination = inclination[inclination[:,-2]>-10,:]
        inclination = inclination[inclination[:,-2]<10,:]
        
        # PLOT MEAN INCLINATION
        
        # create a figure with a single plot and axis
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5.5))
        
        # display the background image
        im0 = ax.imshow(img0,cmap='gray',alpha=0.3)
        
        # plot the points colored with mean inclinations
        sca0 = ax.scatter(inclination[:,0],inclination[:,1],c=inclination[:,-2],cmap=cmap, vmin=-3, vmax=1,s=3)

        # add a colorbar and define its position
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(sca0, cax=cax, label='Mean Inclination []')
        
        # change the resolution of the plot
        plt.gcf().set_dpi(300)
        
        # save the mean inclination image
        plt.savefig(str(out_path.joinpath('000_mean_inclination.jpg')),bbox_inches='tight')
        plt.show()
        
        # PLOT STANDARD DEVIATION OF INCLINATION
        
        # create a figure with a single plot and axis
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5.5))
        
        # display the background image
        im0 = ax.imshow(img0,cmap='gray',alpha=0.3)
        
        # plot the points colored with mean inclinations
        sca0 = ax.scatter(stddev[:,0],stddev[:,1],c=stddev[:,2],cmap=cmap, vmin=0, vmax=3,s=3)

        # add a colorbar and define its position
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1./aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.colorbar(sca0, cax=cax, label='Inclination Standard Deviation []')
        
        # change the resolution of the plot
        plt.gcf().set_dpi(300)
        
        # save the mean inclination image
        plt.savefig(str(out_path.joinpath('000_stddev_inclination.jpg')),bbox_inches='tight')
        plt.show()

    def extract_incli(self,pointname,pointx,pointy):
        """
        extracts the information of a point from inclinations and
        saves them

        Parameters
        ----------
        pointname : string
            name of the point that should be extract (output file name).
        pointx : integer
            x location in pixel units of the point that should be extracted.
        pointy : integer
            y location in pixel units of the point that should be extracted..

        Returns
        -------
        None.

        """
        incli_list = [p for p in self._inpath.joinpath('inclination').iterdir() if p.suffix.lower() == '.csv' and
                     str(p.stem).startswith('incli-')]
        
        incli = np.ones((len(incli_list),))
        
        tempdate=list()
        
        # loop through all 3D results an get the results at the point location
        for ii,fname in enumerate(incli_list):
            try:
                res = np.genfromtxt(str(fname), delimiter=',')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            except:
                res = np.genfromtxt(str(fname), delimiter=';')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            
            results = res[(res[:,0]==pointx)&(res[:,1]==pointy),:]
            
            incli[ii] = results[:,2]
            
            # get the dates of the 3D results name for indexing
            tempdate.append(Path(fname).stem.split('-')[-3]+'-'+
                            Path(fname).stem.split('-')[-2]+'-'+
                            Path(fname).stem.split('-')[-1])
            
        # save temporary results in a pandas dataframe and save it as csv
        pointDF = pd.DataFrame({'dates':tempdate, 'inclination ':incli})
        
        # define exname into output path with pointname
        exname = self._outpath.parent.joinpath('inclination').joinpath('0_inclincation-' + pointname + '.csv')
        
        pointDF.to_csv(str(exname),index=False)
