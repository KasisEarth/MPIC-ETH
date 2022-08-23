# MPIC-ETH

#### Multiple Pairwise Image Correlation (MPIC) approach using state-of-the-art digital image correlation algorithms (FFT, ZNCC, and CXC)

#### Quantification of Displacements of Features within multiple input images and creation of time series of displacement rates and cumulative displacements. The resulting displacement measurements can be converted from measurements in the image plane (2D) to global 3D displacements with three world coordinate matrices (not created with this tool).

#### Processing Steps:

- Preprocessing: Wallis Filter, OR image representation
- DIC: FFT (Bickel et al, 2018) and ZNCC, CXC with LAMMA approach (Dematteis et al. 2022)
- Postprocessing: Threshold Filter, Vector Filter, Arithmetic Mean Filter
- Georectification: Based on input world coordinate matrices
- Time Series Creation: Calculating mean hourly velocities from multiple pairwise image correlation results
