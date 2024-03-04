################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion

    # [Y I Q] = [0.299 0.587 0.114; 0.596 -0.275 -0.321; 0.212 -0.528 0.311] * [R G B]'
    # Y = 0.299 * R + 0.587 * G + 0.114 * B

    img_gray = np.zeros((img_color.shape[0], img_color.shape[1]), dtype = np.float64)

    for i in range(0, img_color.shape[0]):
        for j in range(0, img_color.shape[1]):
            img_gray[i][j] = 0.299 * img_color[i][j][0] + 0.587 * img_color[i][j][1] + 0.114 * img_color[i][j][2]

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    size = int(sigma * (2* np.log(1000))**0.5)
    x = np.arange(-size, size+1)
    filter = np.exp((x ** 2) / -2 / (sigma ** 2))


    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    
    # conv with zero padding
    img_filtered = convolve1d(img, filter, mode='constant', cval = 0.0)
    
    #  Normalization
    ones = np.ones_like(img)
    # convolve the ones with the filter
    img_weight = convolve1d(ones, filter, mode='constant', cval = 0.0)

    # divide the image by the weight
    img_smoothed = np.divide(img_filtered, img_weight)

    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the horizontal direction
    img = smooth1D(img, sigma)

    # TODO: smooth the image along the horizontal direction
    img = smooth1D(img.T, sigma)
    img_smoothed = img.T

    # plt.imshow(np.float32(img_smoothed), cmap = 'gray')

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy
    # gradient(I) = [Ix, Iy] -> Ix = partial(I)/partial(x), Iy = partial(I)/partial(y)

    #  compute Ix and Iy (finite difference method)
    # Ix = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    # Ix = convolve1d(img, [1, 0, -1], axis = 1, mode = 'constant', cval = 0.0)
    # Iy = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    # Iy = convolve1d(img, [1, 0, -1], axis = 0, mode = 'constant', cval = 0.0)

    # compute Ix and Iy using np.gradient
    Ix = np.gradient(img, axis = 1)
    Iy = np.gradient(img, axis = 0)
    

    # TODO: compute Ix2, Iy2 and IxIy

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # TODO: smooth the squared derivatives

    Ix2 = smooth2D(Ix2, sigma)
    Iy2 = smooth2D(Iy2, sigma)
    IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    R = np.zeros((img.shape[0], img.shape[1]), dtype = np.float64)
    k = 0.04
    #  R = lambda1 * lambda2 - k(lambda1 + lambda2)^2
    # Note: lambda1 * lambda2 = det(H), lambda1 + lambda2 = trace(H)
    R = Ix2 * Iy2 - IxIy ** 2 - k * (Ix2 + Iy2) ** 2

    # print(f"R: {R}")

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    # Note:
    # f(x, y) = ax^2 + by^2 + cx + dy + e
    # f(0,0) = e
    # f(-1,0) = a - c + e
    # f(1,0) = a + c + e
    # f(0,-1) = b - d + e
    # f(0,1) = b + d + e
    corners = []
    for i in range(1, R.shape[0] - 1):
        for j in range(1, R.shape[1] - 1):
            eight_neighbours = np.array([R[i - 1][j - 1], R[i - 1][j], R[i - 1][j + 1], R[i][j - 1], R[i][j + 1], R[i + 1][j - 1], R[i + 1][j], R[i + 1][j + 1]])
            if R[i][j] > np.max(eight_neighbours):
                # perform quadratic approximation by formula 
                a = (R[i-1][j] + R[i+1][j] - 2 * R[i][j]) / 2
                b = (R[i][j-1] + R[i][j+1] - 2 * R[i][j]) / 2
                c = (R[i+1][j] - R[i-1][j]) / 2
                d = (R[i][j+1] - R[i][j-1]) / 2
                e = R[i][j]
                if a != 0 and b != 0:
                    i_subpix = i -(c / (2 * a))
                    j_subpix = j -(d / (2 * b))
                    r = e - (c ** 2) / (4 * a) - (d ** 2) / (4 * b)
                    corners.append([j_subpix, i_subpix, r])


    # TODO: perform thresholding and discard weak corners
    print(f"corners: {corners}")
    strong_corners = []
    for i in range(0, len(corners)):
        if corners[i][2] > threshold:
            strong_corners.append(corners[i])
    
    corners = strong_corners

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#  show corner detection result
################################################################################
def show_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    plt.ion()
    fig = plt.figure('Harris corner detection')
    plt.imshow(img_color)
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  save corners to a file
################################################################################
def save_corners(outputfile, corners) :
    # input:
    #    outputfile - path of the output file
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(outputfile, 'w')
        file.write('{}\n'.format(len(corners)))
        for corner in corners :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(corner[0], corner[1], corner[2]))
        file.close()
    except :
        print('Error occurs in writing output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load corners from a file
################################################################################
def load_corners(inputfile) :
    # input:
    #    inputfile - path of the file containing corner detection output
    # return:
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)

    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading {} corners'.format(nc))
        corners = np.zeros([nc, 3], dtype = np.float64)
        for i in range(nc) :
            line = file.readline()
            x, y, r = line.split()
            corners[i] = [np.float64(x), np.float64(y), np.float64(r)]
        file.close()
        return corners
    except :
        print('Error occurs in loading corners from \'{}\'.'.format(inputfile))
        sys.exit(1)

################################################################################
#  main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0,
                        help = 'sigma value for Gaussain filter (default = 1.0)')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6,
                        help = 'threshold value for corner detection (default = 1e6)')
    parser.add_argument('-o', '--output', type = str,
                        help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : {}'.format(args.image))
    print('sigma      : {:.2f}'.format(args.sigma))
    print('threshold  : {:.2e}'.format(args.threshold))
    print('output file: {}'.format(args.output))
    print('------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('{} corners detected...'.format(len(corners)))
    show_corners(img_color, corners)

    # save corners to a file
    if args.output :
        save_corners(args.output, corners)
        print('corners saved to \'{}\'...'.format(args.output))

if __name__ == '__main__' :
    main()
