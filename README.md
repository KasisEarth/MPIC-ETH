# MPIC-ETH

#### Multiple Pairwise Image Correlation (MPIC) approach using state-of-the-art digital image correlation algorithms (FFT, ZNCC, and CXC)

#### Quantification of Displacements of Features within multiple input images and creation of time series of displacement rates and cumulative displacements. The resulting displacement measurements can be converted from measurements in the image plane (2D) to global 3D displacements with three world coordinate matrices (not created with this tool).

#### Processing Steps:

- Preprocessing: Wallis Filter, OR image representation
- DIC: FFT (Bickel et al, 2018) and ZNCC, CXC with LAMMA approach (Dematteis et al. 2022)
- Postprocessing: Threshold Filter, Vector Filter, Arithmetic Mean Filter
- Georectification: Based on input world coordinate matrices
- Time Series Creation: Calculating mean hourly velocities from multiple pairwise image correlation results

## External/integrated functions/algorithms:

1) This routine bases on the basic workflow of DIC_FFT_ETHZ (https://github.com/bickelmps/DIC_FFT_ETHZ/)
	
	Citation: Bickel, V.T.; Manconi, A.; Amann, F. Quantitative Assessment of Digital Image Correlation Methods to Detect and Monitor Surface Displacements of Large Slope Instabilities. Remote Sens. 2018, 10, 865.
	
	Integrated functions/algorithms:
		
	- dftregistration (in dic.py script) by Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008). Parts of their code has been taken from: J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458 (1990). Online: https://de.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation (Status: May 5th, 2018) License included in "DIC/dft_registration_license.txt"
		
	- wallis (in preprocessing.py script), underlying principle taken from Baltsavias, E. P. "Multiphoto geometrically constrained matching." Doctoral Thesis, ETH Zurich, (1991).

2) The implementation of the ZNCC and CXC algorithms base on the LAMMA routine (https://github.com/niccolodematteis/LAMMA/)

	Citation: Dematteis, Niccolò, Daniele Giordan, Bruno Crippa, and Oriol Monserrat. 2022. “Fast Local Adaptive Multiscale Image Matching Algorithm for Remote Sensing Image Correlation.” Computers & Geosciences 159:104988. doi: 10.1016/j.cageo.2021.104988

3) The cropping function for the co-registration (in dic.py script) bases on a feed on Stackoverflow

	Source: Samplebias via Stackoverflow, posted May 26th 2011, retrieved February 2022
		https://stackoverflow.com/questions/6136588/image-cropping-using-python/8696558

## Instructions

1) Place input images into "input" folder (examples are included in "input" folder)

	#### ! Image names must be in the format "YYYYMMDDThhmmss" for time series creation !
	
	- if the 2D displacements should be georectified, place three world coordinate matrices (X,Y,Z) as *.csv files into the "additional_data" folder
	- if the results should be masked, place the mask as *.csv file into the "additional_data" folder


2) Open main file "MPIC" and customize settings:

	- WALLIS : 		if True, a Wallis Filtering is applied
	- REPRESENTATION : 	'in' for IN image representation, 'or' for OR image representation
	- METHOD : 		'fft', 'zncc', 'cxc' for the respective algorithm
	- BAS_MIN / BAS_MAX : 	Minimal and Maximal temporal baseline, n-1 images are skipped in the image pairing.
	- CROPPING : 		if 0, the entire image is used for coregistration
				if 1, a patch of the input image can be used for the coregistration
				if 2, a patch has to be defined in the main file by indexing for the coregistration (called CROPIND = [up,down,left,right]
	- filtertype : 		if 1, error threshold filter
				if 2, magnitude threshold filter
				if 3, arithmetic mean filter
				if 4, vector filter
	- plottype : 		if 1, scatter plot for every pairwise DIC is shown
				if 2, quiver plot for every pairwise DIC is shown
	- georect : 		if True, the time series results are georectified
	- timeser : 		if True, a time series is created
	- ts_filt : 		if True, the time series results are filtered to identify outliers
	- masking : 		if True, the time series results are masked with the mask in the "additional_data" folder


3) Advanced settings of processing steps have to be changed in the corresponding scripts.


	Preprocessing --------------------------------------------------------------------------------------------------------

		- WIN : 	block size of the wallis filter, standard: 32px 
		- TARM : 	target mean value in a block after wallis filtering, standard: 150
		- TARS : 	target standard deviation in block after wallis filtering, standard: 150
		- BRIGHTNESS : brightness enforcing constant of wallis filter
		- CONTRAST : 	contrast enforcing constant of wallis filter

	DIC ------------------------------------------------------------------------------------------------------------------

		- OVSCO : 	oversampling factor for co-registration
		- SPLIT : 	splitting image for co-registration, if = 1 no splitting
		- TILESIZE : 	size of the template in LAMMA (cxc, zncc)
		- NODEDIST : 	smallest grid size (level) in LAMMA (cxc, zncc)
		- OVSLAMMA : 	oversampling factor for LAMMA
		- MAXBAND : 	starting radius around template for cross correlation (search area)
		- MAXSCALE : 	maximal grid size in LAMMA (cxc, zncc)
		- TOLERANCE :  tolerance which is added to the search radius in LAMMA
		- WIN :  	template size in FFT algorithm
		- OVSFFT :  	oversampling in FFT algorithm
		- FFTSTEP : 	step size in the fft algorithm (comparable to NODEDIST)

	Postprocessing ---------------------------------------------------------------------------------------------------------

		- THR : 	error masking threshold, values between 0-1 ( 1=no mask )
		- MAG : 	magnitude masking threshold, 2D displacement magnitudes larger than MAG are filtered out
				when MAG = None --> MAG = half the window size by default.
		- MFWS : 	mean filter window size for arithmettic mean filter
		- CUT : 	the window size divieded by CUT gives the value that will be used to cutDX and DY values that are larger than WIN/CUT
        		CUT = 0 means no cut off
		- PLTMIN : 	minimum 2D displacement magnidute that is visualized in the plotting
		- PLTMAX : 	maximum 2D displacement magnitude that is visualized in the left plot
				Only for visualization, when not defined (set PLTMAX = None), it will be the 10th largest displacement measured
		- VEC1 : 	the window size diveded by VEC1 is the threshold which is used to filter magnitudes in the vector filter
		- VEC2 : 	the window size divided by VEC2 is the threshold which is used to filter DX and DY in the vector filter
		- VEC3 : 	threshold in the vector filter that is used to filter direciton changes in a 3x3 neighbourhood (in rad)
		- MAXDECORR : 	threshold, more decorrelated results in the masked results indicate that the pairing of the images is bad 
				and the results are neglected
		- MASKNAME : 	name of the mask file in the additional_data folder, e.g. 'mask.csv'

	Georectification -------------------------------------------------------------------------------------------------------

		- X_name : 	filename of the X world coordinate matrix saved in the "additional_data" folder
		- Y_name : 	filename of the Y world coordinate matrix saved in the "additional_data" folder
		- Z_name : 	filename of the Z world coordinate matrix saved in the "additional_data" folder


	Time Series ------------------------------------------------------------------------------------------------------------

		- HOURS : 	the number of hours needed in a day to consider it within the time period
		- SINPLEPLOT : 	if True, a single plot for the gif is created
		- MULTIPLOT : 	if True, two plots for the gif are created
		- scale_2d : 	list of minimum and maximum of the scale for the 2D GIF creation    
        			[min velocity, max velocity, min cumulative displacement, max cum. dip.]
		- scale_3d : 	list of minimum and maximum of the scale for the 3D GIF creation    
        			[min velocity, max velocity, min cumulative displacement, max cum. dip.]

	------------------------------------------------------------------------------------------------------------------------
  
4) Execute "MPIC.py" file

5) Collect results in the "output" folder:

	- csv file for every pairwise image matching result
	- csv file for every filtered pairwise image matching result
	- 000_extracted_decorrelation file that summarizes the number of decorrelated points per day from the time series

	- GIF: 		
	    when time series creation (timeser) is enabled
			distribution of velocity and cumulative displacement in 2D and 3D for every day saved as a jpg and merged to a GIF

	- images:
	    when single pairwise image matching results are plotted (plottype = 1 or plottype = 2)
			visualized the results of the single pairwise images in plots
			images from the input folder are saved as greyscale (for GIF creation)

	- inclination:	
	    inclination of the 3D time series displacement results as .csv
			and visualization of inclination as a GIF
			0_inclination-*POINTNAME* files with the extracted inclination measurements for specific points defined in the MPIC.py file

	- time-series:	
	    "hours" folder with the mean displacement measured per hour (=velocity)
			TS-YYYY-MM-DD files with mean displacement per hour for every day
			CUM-TS-YYYY-MM-DD files with the cumulated displacement for every day
			3D-CUM-TS-YYYY-MM-DD files with the georectified cumulated displacement for every day
			3D-extracted-CUM-*POINTNAME* files with extracted cumulative 3D displacement for specific points defined in the MPIC.py file

