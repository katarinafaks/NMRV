import time

import numpy as np
import cv2

# from scipy.signal import convolve2d
from matplotlib import pyplot as plt


from ex1_utils import gaussderiv, gausssmooth, show_flow

def lucas_kanade ( im1 , im2 , N ):
    # im1 − first image matrix (grayscale)
    # im2 − second image matrix (grayscale)
    # n − size of the neighbourhood (N x N)
    # returns: two matrices of the same size as the input images
    # that contain "u" and "v" components of the optical flow
    # displacement vectors

    
    # DELETE ME AFTER TESTS IF NOT NEEDED:
    # im1 = im1.astype(np.float32) / 255.0
    # im2 = im2.astype(np.float32) / 255.0

    #Need: spatial derivatives (Ix, Iy) and temporal derivative (It)
    It = gausssmooth(im2 - im1,1) # temporal derivative
    # Ix, Iy = gaussderiv(im1, 1) # spatial derivatives, sigma predetermined--adjust

    Iavg = (im1 + im2) / 2 # Using average image for spatial derivatives 
    Ix, Iy = gaussderiv(Iavg, 1) # spatial derivatives, sigma predetermined--adjust
    
    # unit kernel of all ones for sums:
    kernel = np.ones((N, N), dtype=np.float32) 
    
    I_x2 = cv2.filter2D(Ix * Ix, -1, kernel)
    I_y2 = cv2.filter2D(Iy * Iy, -1, kernel)
    I_xy = cv2.filter2D(Ix * Iy, -1, kernel)
    I_xt = cv2.filter2D(Ix * It, -1, kernel)
    I_yt = cv2.filter2D(Iy * It, -1, kernel)

    # compute u and v:
    D = I_x2 * I_y2 - (I_xy)**2    

    
    #For reliability of LK, we use Harris response detector:
    k = 0.04 #empirically determined constant (from WIKI: between 0.04 and 0.06)
    trace = I_x2 + I_y2
    R = D - k * trace**2
    mask = R > 1e-5 # threshold for reliable flow, empirically determined
    
    eps = 1e-6
    D[np.abs(D) < eps] = eps
    
    u = (-I_y2 * I_xt + I_xy * I_yt) / D
    v = (-I_x2 * I_yt + I_xy * I_xt) / D
    
    # Apply reliability mask
    u[~mask] = 0
    v[~mask] = 0
    
    return u,v 


def pyramidal_lk(im1, im2, N, num_levels = 3, scale = 0.5):
    # Build pyramids
    pyramid1 = [im1]
    pyramid2 = [im2]
    
    for _ in range(1,num_levels):
        pyramid1.append(cv2.resize(pyramid1[-1], None, fx = scale, fy = scale))
        pyramid2.append(cv2.resize(pyramid2[-1],None, fx = scale, fy = scale))
        #cv2.pyrDown <- alternative, test
    
    u = np.zeros_like(pyramid1[-1], dtype=np.float32)
    v = np.zeros_like(pyramid1[-1], dtype=np.float32)
    
    for lvl in reversed(range(num_levels)):
        im1_lvl = pyramid1[lvl]
        im2_lvl = pyramid2[lvl]
        
        if lvl < num_levels - 1:
            u = cv2.resize(u, (im1_lvl.shape[1], im1_lvl.shape[0])) * (1/scale)
            v = cv2.resize(v, (im1_lvl.shape[1], im1_lvl.shape[0])) * (1/scale)
        
        # warp second image
        h,w = im1_lvl.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + u).astype(np.float32)
        map_y = (grid_y + v).astype(np.float32)
        
        im2_warped = cv2.remap(im2_lvl, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        #compute flow at curr level
        du, dv = lucas_kanade(im1_lvl, im2_warped, N)
        
        #update flow:
        u += du
        v += dv
    return u, v
    

def horn_schunck ( im1 , im2 , n_iters , lmbd , lk_init = False):
    # im1 − first image matrix (grayscale)
    # im2 − second image matrix (grayscale)
    # n_iters − number of iterations (try several hundred)
    # lmbd − parameter
    # returns: two matrices of the same size as the input images
    # that contain "u" and "v" components of the optical flow
    
    # Initialize u and v to zero:
    if lk_init:
        u, v = lucas_kanade(im1, im2, 3)
    else:
        u = np.zeros_like(im1, dtype=np.float32)
        v = np.zeros_like(im1, dtype=np.float32)
    
    # Temporal derivative:
    It = gausssmooth(im2 - im1,1)
    # Spatial derivatives:
    Iavg = (im1 + im2) / 2 # Using average image for spatial derivatives
    Ix, Iy = gaussderiv(Iavg, 1) 
    
    kernel_average = np.asarray([[0, 1/4, 0],
                                [1/4, 0, 1/4],
                                [0, 1/4, 0]], dtype=np.float32)
    
    for i in range(n_iters):
        # averages of u and v
        u_a = cv2.filter2D(u, -1, kernel_average)
        v_a = cv2.filter2D(v, -1, kernel_average)
        
        P = Ix * u_a + Iy * v_a + It
        D = lmbd + Ix**2 + Iy**2
        
        eps = 1e-6
        D[np.abs(D) < eps] = eps
    
        
        # Iterative step
        u = u_a - Ix * P / D
        v = v_a - Iy * P / D
        

    return u, v





def plot_flow_both_func(im1,im2):
    U_lk, V_lk = lucas_kanade(im1, im2, 3)
    U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap='gray')
    ax1_12.imshow(im2, cmap='gray')
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas-Kanade Optical Flow')
    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1, cmap='gray')
    ax2_12.imshow(im2, cmap='gray')
    show_flow(U_hs, V_hs, ax2_21, type='angle')
    show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    fig2.suptitle('Horn-Schunck Optical Flow')
    plt.show()


def time_comparison(im1, im2):
    start_time = time.time()
    lucas_kanade(im1, im2, 3)
    lk_time = time.time() - start_time
    
    start_time = time.time()
    horn_schunck(im1, im2, 1000, 0.5)
    hs_time = time.time() - start_time
    
    start_time = time.time()
    horn_schunck(im1, im2, 1000, 0.5, lk_init=True)
    hs_time_lk = time.time() - start_time
    
    print(f"Lucas-Kanade time: {lk_time:.4f} seconds")
    print(f"Horn-Schunck time: {hs_time:.4f} seconds")
    print(f"Horn-Schunck time with LK initialization: {hs_time_lk:.4f} seconds")

if __name__ == "__main__":
    # print("Lucas-Kanade Algorithm")
    # input tests for Lucas-Kanade
    # im1 = cv2.imread('disparity/office_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('disparity/office_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(im1,im2)

    # im1 = cv2.imread('collision/00000001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('collision/00000005.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(im1,im2)
    
    im1 = cv2.imread('lab2/001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    im2 = cv2.imread('lab2/004.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(im1,im2)



    im1 = np.zeros((100,100), np.float32)
    im1[30:70,30:70] = 1

    im2 = np.roll(im1, shift=15, axis=1)  # large motion
    
    u_lk, v_lk = lucas_kanade(im1, im2, 5)
    u_pyr, v_pyr = pyramidal_lk(im1, im2, 5, num_levels=4, scale=0.5)
    
    
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap='gray')
    ax1_12.imshow(im2, cmap='gray')
    show_flow(u_lk, v_lk, ax1_21, type='angle')
    show_flow(u_lk, v_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas-Kanade Optical Flow')
    plt.show()


    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap='gray')
    ax1_12.imshow(im2, cmap='gray')
    show_flow(u_pyr, v_pyr, ax1_21, type='angle')
    show_flow(u_pyr, v_pyr, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Pyramidal Lucas-Kanade Optical Flow')
    plt.show()

    # time_comparison(im1, im2)
    # u,v = pyramidal_lk(im1, im2, 3, num_levels=4, scale=0.5)
    
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(u, v, ax1_21, type='angle')
    # show_flow(u, v, ax1_22, type='field', set_aspect=True)
    # fig1.suptitle('Pyramidal Lucas-Kanade Optical Flow')
    # plt.show()


    # u,v = lucas_kanade(im1, im2, 3)
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(u, v, ax1_21, type='angle')
    # show_flow(u, v, ax1_22, type='field', set_aspect=True)
    # fig1.suptitle('Lucas-Kanade Optical Flow')
    # plt.show()

    # print("Horn-Schunck Algorithm")
    # input tests for Horn-Schunck
    
    
    
    
    


