################################################################################
# COMP3317 Computer Vision
# Assignment 4 - Camera calibration
################################################################################
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from numpy.linalg import lstsq, qr, inv

################################################################################
#  estimate planar projective transformations for the 2 calibration planes
################################################################################
def calibrate2D(ref3D, ref2D) :
    #  input:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D
    # return:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    numOfCorners = ref3D.shape[0]
    A_xz = np.zeros((numOfCorners * 2, 8), dtype=np.float64)
    A_yz = np.zeros((numOfCorners * 2, 8), dtype=np.float64)
    b = np.zeros((numOfCorners * 2), dtype=np.float64)

    # TODO : form the matrix equation Ap = b for the X-Z plane
    # extract X Y Z from ref3D and u v from ref2D
    X = ref3D[:, 0]
    Y = ref3D[:, 1]
    Z = ref3D[:, 2]
    u = ref2D[:, 0]
    v = ref2D[:, 1]

    # fill in A
    # Eq1: X_i * p_00 + Z_i * p_01 + p_02 - u_i * X_i * p_20 - u_i * Z_i * p_21 = u_i
    # Eq2: X_i * p_10 + Z_i * p_11 + p_12 - v_i * X_i * p_20 - v_i * Z_i * p_21 = v_i
    # Reference: L7 slide 24
    for i in range(numOfCorners//2):    # half of the corners are on the X-Z plane
        A_xz[2*i, :] = [X[i], Z[i], 1, 0, 0, 0, -u[i]*X[i], -u[i]*Z[i]]    # Eq1
        b[2*i] = u[i]
        A_xz[2*i+1, :] = [0, 0, 0, X[i], Z[i], 1, -v[i]*X[i], -v[i]*Z[i]]  # Eq2
        b[2*i+1] = v[i]

    # TODO : solve for the planar projective transformation using linear least squares
    p = np.linalg.lstsq(A_xz, b, rcond=None)[0]
    # set p_22 to 1 (overall scale)
    p = np.append(p, 1)
    Hxz = p.reshape(3, 3)

    # TODO : form the matrix equation Ap = b for the Y-Z plane
    # fill in A
    # Eq1: Y_i * p_00 + Z_i * p_01 + p_02 - u_i * Y_i * p_20 - u_i * Z_i * p_21 = u_i
    # Eq2: Y_i * p_10 + Z_i * p_11 + p_12 - v_i * Y_i * p_20 - v_i * Z_i * p_21 = v_i
    for i in range(numOfCorners//2, numOfCorners):    # half of the corners are on the Y-Z plane
        # i takes values from 4 to 7
        A_yz[2*i, :] = [Y[i], Z[i], 1, 0, 0, 0, -u[i]*Y[i], -u[i]*Z[i]]    # Eq1
        b[2*i] = u[i]
        A_yz[2*i+1, :] = [0, 0, 0, Y[i], Z[i], 1, -v[i]*Y[i], -v[i]*Z[i]]  # Eq2
        b[2*i+1] = v[i]

    # TODO : solve for the planar projective transformation using linear least squares
    p = np.linalg.lstsq(A_yz, b, rcond=None)[0]
    # set p_22 to 1 (overall scale)
    p = np.append(p, 1)
    Hyz = p.reshape(3, 3)

    return Hxz, Hyz

################################################################################
#  generate correspondences for all the corners on the 2 calibration planes
################################################################################
def gen_correspondences(Hxz, Hyz, corners) :
    # input:
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane
    #    corners - a n0 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n0 being the number of corners)
    # return:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D

    # TODO : define 3D coordinates of all the corners on the 2 calibration planes


    cornersXZ_3d = np.ones((80, 3), dtype=np.float64)
    cornersYZ_3d = np.ones((80, 3), dtype=np.float64)
    # calculate the 3D coordinates of the corners on the X-Y calibaration plane
    cornersXZ_3d[:, 0] = np.repeat(np.arange(10) + 0.5, 8)  # X coordinates for X-Y plane
    cornersXZ_3d[:, 1] = np.tile(np.arange(8) + 0.5, 10)    # Y coordinates for X-Y plane
    # calculate the 3D coordinates of the corners on the X-Z calibaration plane
    cornersYZ_3d[:, 0] = np.repeat(np.arange(10) + 0.5, 8)  # X coordinates for X-Z plane
    cornersYZ_3d[:, 1] = np.tile(np.arange(8) + 0.5, 10)    # Z coordinates for X-Z plane

    # TODO : project corners on the calibration plane 1 onto the image
    cornersCalibXZ_3d = []
    cornersCalibXZ_2d = []
    for i in range(cornersXZ_3d.shape[0]):
        if (cornersXZ_3d[i][0] > 0 and cornersXZ_3d[i][0] <= 10
                and cornersXZ_3d[i][1] > 0 and cornersXZ_3d[i][1] <= 8):
            cornersCalibXZ_3d.append(cornersXZ_3d[i][0:2])    # [X, Z]
            homogXZ_2d = np.dot(Hxz, cornersXZ_3d[i])
            homogXZ_2d = homogXZ_2d / homogXZ_2d[2]    # normalize to get [u, v, 1]
            cornersCalibXZ_2d.append(homogXZ_2d[0:2])         # [u, v]

    # convert back to numpy array
    cornersCalibXZ_3d = np.array(cornersCalibXZ_3d)
    # add column of 0s (Y) to cornersCalibXZ_3d
    cornersCalibXZ_3d = np.insert(cornersCalibXZ_3d, 1, 0, axis=1)
    cornersCalibXZ_2d = np.array(cornersCalibXZ_2d)

    # TODO : project corners on the calibration plane 2 onto the image
    cornersCalibYZ_3d = []
    cornersCalibYZ_2d = []
    for i in range(cornersYZ_3d.shape[0]):
        if (cornersYZ_3d[i][0] > 0 and cornersYZ_3d[i][0] <= 10
                and cornersYZ_3d[i][1] > 0 and cornersYZ_3d[i][1] <= 8):
            cornersCalibYZ_3d.append(cornersYZ_3d[i][0:2])  # [Y, Z]
            homogYZ_2d = np.dot(Hyz, cornersYZ_3d[i])
            homogXZ_2d = homogYZ_2d / homogYZ_2d[2]
            cornersCalibYZ_2d.append(homogXZ_2d[0:2])       # [u, v]

    # convert back to numpy array
    cornersCalibYZ_3d = np.array(cornersCalibYZ_3d)
    # add column of 0s (X) to cornersCalibYZ_3d
    cornersCalibYZ_3d = np.insert(cornersCalibYZ_3d, 0, 0, axis=1)
    cornersCalibYZ_2d = np.array(cornersCalibYZ_2d)

    # TODO : locate the nearest detected corners
    ref3D = np.concatenate((cornersCalibXZ_3d, cornersCalibYZ_3d), axis=0)
    # find the nearest corner for each ref2D point
    ref2D = find_nearest_corner(np.concatenate((cornersCalibXZ_2d, cornersCalibYZ_2d), axis=0), corners)

    return ref3D, ref2D

################################################################################
#  estimate the camera projection matrix
################################################################################
def calibrate3D(ref3D, ref2D) :
    # input:
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners in ref3D
    # output:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    numOfCorners = ref3D.shape[0]
    A = np.zeros((numOfCorners * 2, 11), dtype=np.float64)
    b = np.zeros((numOfCorners * 2), dtype=np.float64)

    # TODO : form the matrix equation Ap = b for the camera
    # extract X Y Z from ref3D and u v from ref2D
    X = ref3D[:, 0]
    Y = ref3D[:, 1]
    Z = ref3D[:, 2]
    u = ref2D[:, 0]
    v = ref2D[:, 1]

    # fill in A
    # Eq1: X_i * p_00 + Y_i * p_01 + Z_i * p_02 + p_03 - u_i * X_i * p_20 - u_i * Y_i * p_21 - u_i * Z_i * p_22 = u_i
    # Eq2: X_i * p_10 + Y_i * p_11 + Z_i * p_12 + p_13 - v_i * X_i * p_20 - v_i * Y_i * p_21 - v_i * Z_i * p_22 = v_i
    # Reference: L7 slide 6
    for i in range(numOfCorners):
        A[2*i, :] = [X[i], Y[i], Z[i], 1, 0, 0, 0, 0, -u[i]*X[i], -u[i]*Y[i], -u[i]*Z[i]]
        A[2*i+1, :] = [0, 0, 0, 0, X[i], Y[i], Z[i], 1, -v[i]*X[i], -v[i]*Y[i], -v[i]*Z[i]]
        b[2*i] = u[i]
        b[2*i+1] = v[i]

    # TODO : solve for the projection matrix using linear least squares
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    p = np.append(p, 1)
    P = p.reshape(3, 4)

    return P

################################################################################
#  decompose the camera calibration matrix in K[R T]
################################################################################
def decompose_P(P) :
    # input:
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix
    # output:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    # Reference: L7 slide 19

    # TODO: extract the 3 x 3 submatrix from the first 3 columns of P
    P_sub = P[:, 0:3]

    # TODO : perform QR decomposition on the inverse of [P0 P1 P2]
    q, r = np.linalg.qr(inv(P_sub))

    # TODO : obtain K as the inverse of R
    K = inv(r)

    # TODO : obtain R as the tranpose of Q
    R = q.T

    # TODO : normalize K
    # if k_22 is not 1, divide the whole matrix by k_22
    # if k_00 is negative, multiply 1st column and 1st row of R by -1
    # if k_11 is negative, multiply 2nd column and 2nd row of R by -1
    alpha = K[2, 2]
    K = K / alpha
    if K[0, 0] < 0:
        K[:, 0] = -1 * K[:, 0]
        R[0, :] = -1 * R[0, :]
    if K[1, 1] < 0:
        K[:, 1] = -1 * K[:, 1]
        R[1, :] = -1 * R[1, :]

    # TODO : obtain T from P3
    # Formula: T = 1/alpha K^-1 P3
    P3 = P[:, 3]
    T = (1/alpha) * np.dot(np.linalg.inv(K), P3)

    # combine R and T to form RT
    T = T.reshape(3, 1)
    RT = np.hstack((R, T))

    return K, RT

################################################################################
#  check the planar projective transformations for the 2 calibration planes
################################################################################
def check_H(img_color, Hxz, Hyz) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    Hxz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the X-Z plane
    #    Hyz - a 3 x 3 numpy ndarray holding the planar projective transformation
    #          for the Y-Z plane

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # define 3D coordinates of all the corners on the 2 calibration planes
    X_ = np.arange(10) + 0.5 # Y == X
    Z_ = np.arange(8) + 0.5
    X_ = np.tile(X_, 8)
    Z_ = np.repeat(Z_, 10)
    X = np.vstack((X_, Z_, np.ones(80)))

    # project corners on the calibration plane 1 onto the image
    w = Hxz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    # project corners on the calibration plane 2 onto the image
    w = Hyz @ X
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)

    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the 2D correspondences for the 2 calibration planes
################################################################################
def check_correspondences(img_color, ref2D) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref2D - a 160 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of all the corners on the 2 calibration planes

    # plot the image and the correspondences
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(ref2D[:, 0], ref2D[:, 1], 'bx', markersize = 5)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  check the estimated camera projection matrix
################################################################################
def check_P(img_color, ref3D, P) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    ref3D - a 160 x 3 numpy ndarray holding the 3D coodinates of all the corners
    #            on the 2 calibration planes
    #    P - a 3 x 4 numpy ndarray holding the camera projection matrix

    # plot the image
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)

    # project the reference 3D points onto the image
    w = P @ np.append(ref3D, np.ones([len(ref3D), 1]), axis = 1).T
    u = w[0, :] / w[2, :]
    v = w[1, :] / w[2, :]
    plt.plot(u, v, 'r.', markersize = 3)
    plt.show()
    plt.ginput(n = 1, timeout = - 1)
    plt.close(fig)

################################################################################
#  pick seed corners for camera calibration
################################################################################
def pick_corners(img_color, corners) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)
    #    corners - a n x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n being the number of corners)
    # return:
    #    ref3D - a 8 x 3 numpy ndarray holding the 3D coodinates of the extreme
    #            corners on the 2 calibration planes
    #    ref2D - a 8 x 2 numpy ndarray holding the 2D coordinates of the projections
    #            of the corners in ref3D

    # plot the image and corners
    plt.ion()
    fig = plt.figure('Camera calibration')
    plt.imshow(img_color)
    plt.plot(corners[:,0], corners[:,1],'r+', markersize = 5)
    plt.show()

    # define 3D coordinates of the extreme corners on the 2 calibration planes
    ref3D = np.array([(9.5, 0, 7.5), (0.5, 0, 7.5), (9.5, 0, 0.5), (0.5, 0, 0.5),
                      (0, 0.5, 7.5), (0, 9.5, 7.5), (0, 0.5, 0.5), (0, 9.5, 0.5)],
                      dtype = np.float64)
    ref2D = np.zeros([8, 2], dtype = np.float64)
    for i in range(8) :
        selected = False
        while not selected :
            # ask user to pick the corner on the image
            print('please click on the image point for ({}, {}, {})...'.format(
                  ref3D[i, 0], ref3D[i, 1], ref3D[i, 2]))
            plt.figure(fig.number)
            pt = plt.ginput(n = 1, timeout = - 1)
            # locate the nearest detected corner
            pt = find_nearest_corner(np.array(pt), corners)
            if pt[0, 0] > 0 :
                plt.figure(fig.number)
                plt.plot(pt[:, 0], pt[:, 1], 'bx', markersize = 5)
                ref2D[i, :] = pt[0]
                selected = True
            else :
                print('cannot locate detected corner in the vicinity...')
    plt.close(fig)

    return ref3D, ref2D

################################################################################
#  find nearest corner
################################################################################
def find_nearest_corner(pts, corners) :
    # input:
    #    pts - a n0 x 2 numpy ndarray holding the coordinates of the points
    #          (n0 being the number of points)
    #    corners - a n1 x 3 numpy ndarray holding the coordinates and strengths
    #              of the detected corners (n1 being the number of corners)
    # return:
    #    selected - a n0 x 2 numpy ndarray holding the coordinates of the nearest_corner
    #               corner

    x = corners[:, 0]
    y = corners[:, 1]
    x_ = pts[:, 0]
    y_ = pts[:, 1]

    # compute distances between the corners and the pts
    dist = np.sqrt(np.square(x.reshape(-1,1).repeat(len(x_), axis = 1) - x_)
                 + np.square(y.reshape(-1,1).repeat(len(y_), axis = 1) - y_))
    # find the index of the corner with the min distance for each pt
    min_idx = np.argmin(dist, axis = 0)
    # find the min distance for each pt
    min_dist = dist[min_idx, np.arange(len(x_))]
    # extract the nearest corner for each pt
    selected = corners[min_idx, 0:2]
    # identify corners with a min distance > 10 and replace them with [-1, -1]
    idx = np.where(min_dist > 10)
    selected[idx, :] = [-1 , -1]
    return selected

################################################################################
#  save K[R T] to a file
################################################################################
def save_KRT(outputfile, K, RT) :
    # input:
    #    outputfile - path of the output file
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(outputfile, 'w')
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e}\n'.format(K[i,0], K[i, 1], K[i, 2]))
        for i in range(3) :
            file.write('{:.6e} {:.6e} {:.6e} {:.6e}\n'.format(RT[i, 0], RT[i, 1],
                       RT[i, 2], RT[i, 3]))
        file.close()
    except :
        print('Error occurs in writing output to \'{}\'.'.format(outputfile))
        sys.exit(1)

################################################################################
#  load K[R T] from a file
################################################################################
def load_KRT(inputfile) :
    # input:
    #    inputfile - path of the file containing K[R T]
    # return:
    #    K - a 3 x 3 numpy ndarry holding the K matrix
    #    RT - a 3 x 4 numpy ndarray holding the rigid body transformation

    try :
        file = open(inputfile, 'r')
        K = np.zeros([3, 3], dtype = np.float64)
        RT = np.zeros([3, 4], dtype = np.float64)
        for i in range(3) :
            line = file.readline()
            e0, e1, e2 = line.split()
            K[i] = [np.float64(e0), np.float64(e1), np.float64(e2)]
        for i in range(3) :
            line = file.readline()
            e0, e1, e2, e3 = line.split()
            RT[i] = [np.float64(e0), np.float64(e1), np.float64(e2), np.float64(e3)]
        file.close()
    except :
        print('Error occurs in loading K[R T] from \'{}\'.'.format(inputfile))
        sys.exit(1)

    return K, RT

################################################################################
#  load image from a file
################################################################################
def load_image(inputfile) :
    # input:
    #    inputfile - path of the image file
    # return:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image (h, w being the height and width of the image)

    try :
        img_color = plt.imread(inputfile)
        return img_color
    except :
        print('Cannot open \'{}\'.'.format(inputfile))
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
        # print('loading {} corners'.format(nc))
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
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 4')
    parser.add_argument('-i', '--image', type = str, default = 'grid1.jpg',
                        help = 'filename of input image')
    parser.add_argument('-c', '--corners', type = str, default = 'grid1.crn',
                        help = 'filename of corner detection output')
    parser.add_argument('-o', '--output', type = str, default='grid1.cam',
                        help = 'filename for outputting camera calibration result')
    args = parser.parse_args()

    print('-------------------------------------------')
    print('COMP3317 Assignment 4 - Camera calibration')
    print('input image : {}'.format(args.image))
    print('corner list : {}'.format(args.corners))
    print('output file : {}'.format(args.output))
    print('-------------------------------------------')

    # load the image
    img_color = load_image(args.image)
    print('\'{}\' loaded...'.format(args.image))

    # load the corner detection result
    corners = load_corners(args.corners)
    print('{} corners loaded from \'{}\'...'.format(len(corners), args.corners))

    # pick the seed corners for camera calibration
    print('pick seed corners for camera calibration...')
    ref3D, ref2D = pick_corners(img_color, corners)

    # estimate planar projective transformations for the 2 calibration planes
    print('estimate planar projective transformations for the 2 calibration planes...')
    H1, H2 = calibrate2D(ref3D, ref2D)
    check_H(img_color, H1, H2)

    # generate correspondences for all the corners on the 2 calibration planes
    print('generate correspondences for all the corners on the 2 calibration planes...')
    ref3D, ref2D = gen_correspondences(H1, H2, corners)
    check_correspondences(img_color, ref2D)

    # estimate the camera projection matrix
    print('estimate the camera projection matrix...')
    P = calibrate3D(ref3D, ref2D)
    print('P = ')
    print(P)
    check_P(img_color, ref3D, P)

    # decompose the camera projection matrix into K[R T]
    print('decompose the camera projection matrix...')
    K, RT = decompose_P(P)
    print('K =')
    print(K)
    print('[R T] =')
    print(RT)
    check_P(img_color, ref3D, K @ RT)

    # save K[R T] to a file
    if args.output :
        save_KRT(args.output, K, RT)
        print('K[R T] saved to \'{}\'...'.format(args.output))

if __name__ == '__main__':
    main()
