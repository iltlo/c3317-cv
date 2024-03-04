# COMP3317 Computer Vision
## Corner Detection

This program performs Harris corner detection on an input image using Python. The main features implemented include:

### 1. RGB to Grayscale Conversion:
   - Converts the input color image to grayscale using the Y channel of the YIQ model.

### 2. 1D Smoothing:
   - Utilizes a 1D horizontal Gaussian filter for smoothing the image.

### 3. 2D Smoothing:
   - Applies 1D convolutions along both horizontal and vertical directions for 2D smoothing.

### 4. Harris Corner Detection:
   - Computes gradients in both x and y directions using finite differences or np.gradient.
   - Computes the squared derivatives (Ix^2, Iy^2, IxIy) and smooths them.
   - Calculates the corner response function R and identifies local maxima as corner candidates.
   - Performs quadratic approximation to localize corners with sub-pixel accuracy.
   - Applies thresholding to discard weak corners.

### 5. Visualization:
   - Displays the detected corners overlaid on the input image.

### 6. Command Line Interface:
   - Supports command-line arguments for specifying input image file, sigma value for Gaussian filter, threshold value for corner detection, and output file for saving corner detection results.

## Usage:
To run the program, execute the script "corner_detection.py" with the following command-line arguments:

```bash
python corner_detection.py -i <input_image_file> -s <sigma_value> -t <threshold_value> -o <output_file>
