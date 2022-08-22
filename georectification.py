# -*- coding: utf-8 -*-
"""
Multiple Pairwise Image Correlation Routine

Based on the FFT DIC routine by Bickel et al. 2018

Author: Lukas Frei
        Master Project D-ERDW ETHZ 2022
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


class Georect():
    '''
    A class for georectification of cumulative displacement results of time series results
    
    ...
    
    Attributes
    ----------
    X_name : string
        filename of the X world coordinate matrix in the additional_data folder
        
    Y_name : string
        filename of the Y world coordinate matrix in the additional_data folder
        
    Z_name : string
        filename of the Z world coordinate matrix in the additional_data folder
        
    '''
    # name must match in georect_data folder.
    X_name = 'X.csv'
    Y_name = 'Y.csv'
    Z_name = 'Z.csv'
    
    def __init__(self,geopath,outpath):

        # path to the output folder. There we get the DIC results
        self._outpath = outpath
        
        # matrices of 3D coorindates at the pixel locations
        print('--- getting 3D coordinates matrices')
        if np.genfromtxt(str(geopath.joinpath(self.X_name)),delimiter=',').shape[1] > 1:
            self._X = np.genfromtxt(str(geopath.joinpath(self.X_name)),delimiter=',')
            self._Y = np.genfromtxt(str(geopath.joinpath(self.Y_name)),delimiter=',')
            self._Z = np.genfromtxt(str(geopath.joinpath(self.Z_name)),delimiter=',')
        else: 
            self._X = np.genfromtxt(str(geopath.joinpath(self.X_name)),delimiter=';')
            self._Y = np.genfromtxt(str(geopath.joinpath(self.Y_name)),delimiter=';')
            self._Z = np.genfromtxt(str(geopath.joinpath(self.Z_name)),delimiter=';')
            
        print(self._X.shape)

        self._input_filt = [p for p in outpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).startswith('filt')]
        self._input_ts = [p for p in outpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).lower().startswith('ts')]
        self._input_cum = [p for p in outpath.iterdir() if p.suffix.lower() == '.csv' and str(p.stem).lower().startswith('cum-ts')]
        self._input_list = self._input_cum  #+ self._input_ts

    def rectify(self):
        print('--- rectify DIC outputs')
        # create meshgrid with size of input coordinate matrices
        x_values = np.linspace(0,self._X.shape[1]-1,self._X.shape[1])
        y_values = np.linspace(0,self._X.shape[0]-1,self._X.shape[0])
        x,y = np.meshgrid(x_values,y_values)
        points = np.empty((x.flatten().size,2))
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        
        
        for fname in self._input_list:

            # import results from DIC
            try:
                res = np.genfromtxt(str(fname), delimiter=',')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            except:
                res = np.genfromtxt(str(fname), delimiter=';')[1:,:] # columns: index, x,y,dx,dy,d2d,error/similarity measure
            # # first change the x and y variables becaused I messed them up in the code
            # temp = copy.copy(res[:,0])
            # res[:,0] = copy.copy(res[:,1])
            # res[:,1] = temp

            # temp = copy.copy(res[:,2])
            # res[:,2] = copy.copy(res[:,3])
            # res[:,3] = temp
            
            # get 3D values per row in the results res
            res3D = np.empty((res.shape[0],10))
            
            # array of dx in integer format to test for decimal results
            testres = np.array(res[:,2],dtype=int)
            
            # absolute difference between testres and dx results
            testres = abs(res[:,2]-testres)
            
            if any(testres > 0):
                # if there are subpixel displacement results, use interpolation
                # of the coordinate matrices to get real world coordinates
                

                
                # empty array to save temporarily the locations
                pointi = np.empty((9,2))
                
                # get the maximum of the array to identify border measurements
                maxx,maxy = self._X.shape
                for ii in range(res.shape[0]):
                    if res[ii,0] == 1120 and res[ii,1] == 640:
                        hoi = 3
                    if res[ii,0] < 2 or res[ii,1] < 2 or res[ii,0] > maxy-2 or res[ii,1] > maxx-2:
                        # if pixel units at border of image, set measurements to nan
                        res3D[ii,0] = self._X[int(res[ii,1]),int(res[ii,0])]
                        res3D[ii,1] = self._Y[int(res[ii,1]),int(res[ii,0])]
                        res3D[ii,2] = self._Z[int(res[ii,1]),int(res[ii,0])]
                        res3D[ii,3] = np.nan
                        res3D[ii,4] = np.nan
                        res3D[ii,5] = np.nan
                        res3D[ii,6] = np.nan
                        res3D[ii,7] = np.nan
                        res3D[ii,8] = np.nan
                        res3D[ii,9] = np.nan
                    
                    else:
                        if any(np.isnan(res[ii,:])):
                            # if there are nan values in the DIC results, no 
                            # rectification is applied
                            res3D[ii,0] = np.nan
                            res3D[ii,1] = np.nan
                            res3D[ii,2] = np.nan
                            res3D[ii,3] = np.nan
                            res3D[ii,4] = np.nan
                            res3D[ii,5] = np.nan
                            res3D[ii,6] = np.nan
                            res3D[ii,7] = np.nan
                            res3D[ii,8] = np.nan
                            res3D[ii,9] = np.nan

                        else:
                            # 3x3 matrix of uv coordinates around the start uv values
                            xi = x[int(res[ii,1])-1:int(res[ii,1])+2,int(res[ii,0])-1:int(res[ii,0])+2]
                            yi = y[int(res[ii,1])-1:int(res[ii,1])+2,int(res[ii,0])-1:int(res[ii,0])+2]
                            
                            # 3x3 matrix to a vector
                            pointi[:,0] = xi.flatten()
                            pointi[:,1] = yi.flatten()
                            
                            # 3x3 matrix of X,Y,Z at same location
                            Xi = self._X[int(res[ii,1])-1:int(res[ii,1])+2,int(res[ii,0])-1:int(res[ii,0])+2]
                            Yi = self._Y[int(res[ii,1])-1:int(res[ii,1])+2,int(res[ii,0])-1:int(res[ii,0])+2]
                            Zi = self._Z[int(res[ii,1])-1:int(res[ii,1])+2,int(res[ii,0])-1:int(res[ii,0])+2]
                            
                            #interpolate the values at the location of uv (potentially subpixel values)
                            res3D[ii,0] = griddata(pointi,Xi.flatten(),res[ii,0:2])
                            res3D[ii,1] = griddata(pointi,Yi.flatten(),res[ii,0:2])
                            res3D[ii,2] = griddata(pointi,Zi.flatten(),res[ii,0:2])

                            try:
                                # 3x3 matrix of uv coordinates around the end uv values
                                xii = x[int(res[ii,1])-1+int(res[ii,3]):int(res[ii,1])+2+int(res[ii,3]),
                                       int(res[ii,0])-1+int(res[ii,2]):int(res[ii,0])+2+int(res[ii,2])]
                                yii = y[int(res[ii,1])-1+int(res[ii,3]):int(res[ii,1])+2+int(res[ii,3]),
                                       int(res[ii,0])-1+int(res[ii,2]):int(res[ii,0])+2+int(res[ii,2])]
                                
                                # 3x3 matrix to a vector
                                pointi[:,0] = xii.flatten()
                                pointi[:,1] = yii.flatten()
                                
                                # 3x3 matrix of X,Y,Z at same location
                                Xii = self._X[int(res[ii,1])-1+int(res[ii,3]):int(res[ii,1])+2+int(res[ii,3]),
                                       int(res[ii,0])-1+int(res[ii,2]):int(res[ii,0])+2+int(res[ii,2])]
                                Yii = self._Y[int(res[ii,1])-1+int(res[ii,3]):int(res[ii,1])+2+int(res[ii,3]),
                                       int(res[ii,0])-1+int(res[ii,2]):int(res[ii,0])+2+int(res[ii,2])]
                                Zii = self._Z[int(res[ii,1])-1+int(res[ii,3]):int(res[ii,1])+2+int(res[ii,3]),
                                       int(res[ii,0])-1+int(res[ii,2]):int(res[ii,0])+2+int(res[ii,2])]
                                
                                 #interpolate the values at the location of uv (potentially subpixel values)
                                res3D[ii,3] = griddata(pointi, Xii.flatten(), res[ii,0:2]+res[ii,2:4])
                                res3D[ii,4] = griddata(pointi, Yii.flatten(), res[ii,0:2]+res[ii,2:4])
                                res3D[ii,5] = griddata(pointi, Zii.flatten(), res[ii,0:2]+res[ii,2:4])
                                
                                # calculate displacements in 3D and total displacement
                                res3D[ii,6] = res3D[ii,3]-res3D[ii,0]
                                res3D[ii,7] = res3D[ii,4]-res3D[ii,1]
                                res3D[ii,8] = res3D[ii,5]-res3D[ii,2]
                                res3D[ii,9] = np.sqrt(res3D[ii,6]**2+res3D[ii,7]**2+res3D[ii,8]**2)
                            except:
                                res3D[ii,3] = np.nan
                                res3D[ii,4] = np.nan
                                res3D[ii,5] = np.nan
                                res3D[ii,6] = np.nan
                                res3D[ii,7] = np.nan
                                res3D[ii,8] = np.nan
                                res3D[ii,9] = np.nan
                         
            
            
            else:
                # No subpixel accuracy, only extract from coordinate matrices
                # input into res3D: X1,Y1.Z1, X2,Y2,Z2, DX/DY/DZ, D3D
                for ii in range(res.shape[0]):
                    if any(np.isnan(res[ii,:])):
                        res3D[ii,0] = np.nan
                        res3D[ii,1] = np.nan
                        res3D[ii,2] = np.nan
                        res3D[ii,3] = np.nan
                        res3D[ii,4] = np.nan
                        res3D[ii,5] = np.nan
                        res3D[ii,6] = np.nan
                        res3D[ii,7] = np.nan
                        res3D[ii,8] = np.nan
                        res3D[ii,9] = np.nan
                    else:
                        # if int(res[ii,3]) > 10:
                        if int(res[ii,0]) == 1440 and int(res[ii,1]) == 416:
                            hoi = 4 #only to set a fix point here to debug
                        try:
                            # get XYZ info from matrices at the pixel location uv and u+du,v+dv
                            res3D[ii,0] = self._X[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,1] = self._Y[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,2] = self._Z[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,3] = self._X[int(res[ii,1])+int(res[ii,3]),int(res[ii,0])+int(res[ii,2])]
                            res3D[ii,4] = self._Y[int(res[ii,1])+int(res[ii,3]),int(res[ii,0])+int(res[ii,2])]
                            res3D[ii,5] = self._Z[int(res[ii,1])+int(res[ii,3]),int(res[ii,0])+int(res[ii,2])]
                            
                            # calculate displacements in 3D and total displacement
                            res3D[ii,6] = res3D[ii,3]-res3D[ii,0]
                            res3D[ii,7] = res3D[ii,4]-res3D[ii,1]
                            res3D[ii,8] = res3D[ii,5]-res3D[ii,2]
                            res3D[ii,9] = np.sqrt(res3D[ii,6]**2+res3D[ii,7]**2+res3D[ii,8]**2)
                        except:
                            res3D[ii,0] = self._X[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,1] = self._Y[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,2] = self._Z[int(res[ii,1]),int(res[ii,0])]
                            res3D[ii,3] = np.nan
                            res3D[ii,4] = np.nan
                            res3D[ii,5] = np.nan
                            res3D[ii,6] = np.nan
                            res3D[ii,7] = np.nan
                            res3D[ii,8] = np.nan
                            res3D[ii,9] = np.nan
            
            res3D = pd.DataFrame({'u':res[:,0], 'v':res[:,1], 'dx':res[:,2], 'dy':res[:,3], 
                                  '2d2':(res[:,3]**2+res[:,2]**2)**0.5,
                                  'X1':res3D[:,0],'Y1':res3D[:,1],'Z1':res3D[:,2],
                                  'X2':res3D[:,3],'Y2':res3D[:,4],'Z2':res3D[:,5],
                                  'dX':res3D[:,6],'dY':res3D[:,7],'dZ':res3D[:,8],'d3D':res3D[:,9],
                                  'max error':res[:,4]})
            
            # save the resulting pandas DataFrame to csv
            res3D.to_csv(str(fname.parent) + '/3D-' + str(fname.name),index=False)