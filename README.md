# MPIC-ETH

#### Multiple Pairwise Image Correlation (MPIC) approach using state-of-the-art digital image correlation algorithms (FFT, ZNCC, and CXC)

#### Processing Steps:

- Preprocessing: Wallis Filter, OR image representation
- DIC: FFT (Bickel et al, 2018) and ZNCC, CXC with LAMMA approach (Dematteis et al. 2022)
- Postprocessing: Threshold Filter, Vector Filter, Arithmetic Mean Filter
- Georectification: Based on input world coordinate matrices
- Time Series Creation: Calculating mean hourly velocities from multiple pairwise image correlation results
