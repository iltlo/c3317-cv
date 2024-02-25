import cv2
import numpy as np

# Load images
image_with_robot = cv2.imread('with_robot.jpg')
image_without_robot = cv2.imread('without_robot.jpg')

# Convert images to grayscale
gray_with_robot = cv2.cvtColor(image_with_robot, cv2.COLOR_BGR2GRAY)
gray_without_robot = cv2.cvtColor(image_without_robot, cv2.COLOR_BGR2GRAY)

# Step1: Calculate absolute difference / difference
diff = cv2.absdiff(gray_with_robot, gray_without_robot)
# diff = cv2.subtract(gray_with_robot, gray_without_robot)

# Display the difference
cv2.imshow('Difference', diff)

# Rescale the difference image by g'(x,y)={g(x,y)-min[g(x,y)]}*(L-1)/{max[g(x,y)]-min[g(x,y)]} for L=255
# diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# Apply rescaling to the difference image
# diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
# cv2.imshow('Rescaled Difference', diff)

# Step 2: Apply thresholding to create binary mask
threshold = 30
_, temp_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
# Invert the mask to create ROI
Mask = cv2.bitwise_not(temp_mask)
cv2.imshow('Mask', Mask)

# Step 3: Perform OR operation to combine ROI and white background
result = cv2.bitwise_or(Mask, gray_with_robot)

# Display or save the result
cv2.imshow('Extracted Robot', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
