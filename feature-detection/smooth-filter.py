import numpy as np
from scipy.ndimage import convolve1d

img = [[12, 16, 18], [13, 11, 30], [15, 15, 16]]
print(img)
sigma = 1

def smooth1D(img, sigma):
  size = int(sigma * (2* np.log(1000))**0.5)
  # print(size)
  x = np.arange(-size, size+1)
  # print(x)
  filter = np.exp((x ** 2) / -2 / (sigma ** 2))
  # filter = [1,1,1]
  # print(filter)
  img_filtered = convolve1d(img, filter, mode='constant')
  print(f"img_filtered: {img_filtered}")
  
  #  Normalization

  # create a matrix with the same shape as the image, all ones
  ones = np.ones_like(img)
  # convolve the ones with the filter
  img_weight = convolve1d(ones, filter, mode='constant')
  print(f"img_weight: {img_weight}")

  # divide the image by the weight
  normalized_img = np.divide(img_filtered, img_weight)

  print(f"img_filtered after normalization: {normalized_img}")
  return normalized_img



def smooth2D(img, sigma):
  # img = np.array(img)
  img = smooth1D(img, sigma)
  img = smooth1D(img.T, sigma)
  img = img.T
  print(img)


smooth2D(img, sigma)





