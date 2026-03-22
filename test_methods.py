from of_methods import lucas_kanade, horn_schunck, pyramidal_lk
from ex1_utils import show_flow
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time


def plot_flow_both_func(im1,im2,window_size=3, lambda_=0.5, num_iters=1000,harris=False,lk_init=False):
    U_lk, V_lk = lucas_kanade(im1, im2, window_size,harris)
    U_hs, V_hs = horn_schunck(im1, im2, num_iters, lambda_, lk_init)
    

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
    
    return lk_time, hs_time, hs_time_lk
    
    

def plot_flow_pyramidal(im1, im2, N, num_levels=3, scale=0.5):
    U_pyr, V_pyr = pyramidal_lk(im1, im2, N, num_levels=num_levels, scale=scale)
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
    ax11.imshow(im1, cmap='gray')
    ax12.imshow(im2, cmap='gray')
    show_flow(U_pyr, V_pyr, ax21, type='angle')
    show_flow(U_pyr, V_pyr, ax22, type='field', set_aspect=True)
    fig.suptitle('Pyramidal Lucas-Kanade Optical Flow')
    plt.show()
    
def plot_flow_lk(im1,im2,window_size=3,harris=False):
    U_lk, V_lk = lucas_kanade(im1, im2, window_size,harris)
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap='gray')
    ax1_12.imshow(im2, cmap='gray')
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle(f'Lucas-Kanade Optical Flow with {window_size=}')    
    plt.show()  
    
def plot_flow_hs(im1,im2,lambda_,lk_init=False):
    U_hs, V_hs = horn_schunck(im1, im2, 20, lmbd = lambda_, lk_init=lk_init)
    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1, cmap='gray')
    ax1_12.imshow(im2, cmap='gray')
    
    show_flow(U_hs, V_hs, ax1_21, type='angle_magnitude')
    show_flow(U_hs, V_hs, ax1_22, type='field', set_aspect=True)
    fig1.suptitle(f'Horn-Schunck Optical Flow with {lambda_=}')    
    plt.show()  
    
    
def iterative_lk(im1,im2,N,iterations=10, harris=False, runtime_test=False):
    
    u,v = np.zeros_like(im1, dtype=np.float32), np.zeros_like(im1, dtype=np.float32)
    
    h,w = im1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    for i in range(iterations):
        # warp im1 towards im2 using current u,v
        map_x = (grid_x + u).astype(np.float32)
        map_y = (grid_y + v).astype(np.float32)
        im1_warped = cv2.remap(im1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        du,dv = lucas_kanade(im1_warped, im2, N, harris)
        
        # update the total flow
        u = u + du
        v = v + dv
    return u,v
    
    
    
        
def images_for_second_part():
    
    
    im1 = cv2.imread('lab2/001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    im2 = cv2.imread('lab2/004.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    
    Ulk, Vlk = lucas_kanade(im1, im2, 3, harris=True)
    Uhs, Vhs = horn_schunck(im1, im2, 1000, lmbd = 10, lk_init=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    # alpha = 0.5
    # beta = 0.5
    # gamma = 0
    # overlay = cv2.addWeighted(im1, alpha, im2, beta, gamma)
    
    h,w = im1.shape
    overlay = np.zeros((h,w,3), dtype=np.uint8)
    overlay[:,:,0] = im1
    overlay[:,:,1] = im2
    overlay[:,:,2] = im2

    ax1.imshow(overlay)
    show_flow(Ulk, Vlk, ax2, type='field', set_aspect=True)
    show_flow(Uhs, Vhs, ax3, type='field', set_aspect=True)
    fig.suptitle('Optical flow field: Lucas-Kanade and Horn-Schunck methods')
    plt.show()
    
    
    
    # plot_flow_pyramidal(im1, im2, N=9, num_levels=4, scale=0.5)
    
    # im1 = cv2.imread('hot_air_ballon.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = np.roll(im1, shift=50, axis=1)  # large motion
    
    # plot_flow_lk(im1, im2, window_size=3, harris=True)
    # plot_flow_pyramidal(im1, im2, N=3, num_levels=4, scale=0.5)
    
    # avg_lk = 0
    # avg_hs = 0
    # avg_hs_lk = 0
    # for i in range(1, 8):
    #     im1 = cv2.imread(f'lab2/0{i}7.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    #     im2 = cv2.imread(f'lab2/0{i+1}0.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    #     t1,t2,t3 = time_comparison(im1, im2)
    #     avg_lk += t1
    #     avg_hs += t2
    #     avg_hs_lk += t3
        
    # print(f"Average Lucas-Kanade time: {avg_lk/7:.4f} seconds")
    # print(f"Average Horn-Schunck time: {avg_hs/7:.4f} seconds")
    # print(f"Average Horn-Schunck time with LK initialization: {avg_hs_lk/7:.4f} seconds")
    
    # im1 = cv2.imread(f'lab2/030.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread(f'lab2/040.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # # row, col = im1.shape[:2]
    # # M = np.float32([[1, 0, 30], [0, 1, 30]])  # shift right by 40 pixels
    # # im2 = cv2.warpAffine(im1, M, (col, row))
    
    
    # u,v,err = horn_schunck(im1, im2, 50, lmbd = 20, lk_init=True, runtime_test=True)
    # u2,v2,err2 = horn_schunck(im1, im2, 50, lmbd = 20, lk_init=False, runtime_test=True)
    
    # # plot_flow_hs(im1, im2, lambda_=2.5, lk_init=True)
    # # plot_flow_hs(im1, im2, lambda_=2.5, lk_init=False)
        
    # plt.clf()
    # plt.yscale('log')
    # plt.plot(err, label='HS with LK initialization')
    # plt.plot(err2, label='HS without LK initialization')
    # plt.legend()
    # plt.xlabel('Iteration')
    # plt.ylabel('Total energy (E_data + lambda * E_smooth) (log scale)')
    # plt.suptitle('Error convergence of Horn-Schunck method')
    # plt.show()
    
    
    # im1 = cv2.imread('collision/00000140.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('collision/00000150.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_lk(im1, im2, window_size=9, harris=True)
    # plot_flow_pyramidal(im1, im2, N=9, num_levels=4, scale=0.5)
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(im1, cmap='gray')
    # ax[1].imshow(im2, cmap='gray')    
    # plt.show()
    

    # u,v = iterative_lk(im1, im2, N=9, iterations=5, harris=True)
    # up,vp = pyramidal_lk(im1, im2, N=9, num_levels=4, scale=0.5)
    # ubasic,vbasic = lucas_kanade(im1, im2, 9, harris=True)
    # fig1, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(10,3))
    # show_flow(u, v, ax1, type='angle_magnitude')
    # show_flow(up, vp, ax2, type='angle_magnitude')
    # show_flow(ubasic, vbasic, ax3, type='angle_magnitude')
    # fig1.suptitle('Angle magnitude of iterative LK, pyramidal LK and basic LK')
    # plt.show()
    
    
    
    
    # Window size = 9, lambda = 3, num_iters = 1000
    # translation1 = cv2.imread('disparity/office2_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # translation2 = cv2.imread('disparity/office2_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(translation1, translation2,window_size=9, lambda_=2, num_iters=1000,harris=False, lk_init=True)
    
    
    # im1 = cv2.imread('lab2/087.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('lab2/090.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_hs(im1, im2, lambda_=0.05, lk_init=True)
    # plot_flow_hs(im1, im2, lambda_=5.5, lk_init=True)


    # im1 = cv2.imread('disparity/cporta_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('disparity/cporta_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0

    # U_lk , V_lk = lucas_kanade(im1, im2, 3, harris = True)
    # U_lk2, V_lk2 = lucas_kanade(im1, im2, 3, harris = False)
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(U_lk, V_lk, ax1_21, type='magnitude')
    # show_flow(U_lk2, V_lk2, ax1_22, type='magnitude')
    # fig1.suptitle(f'Magnitude of LK flow with and without Harris corner detection')
    # plt.show()
    
    # U_hs, V_hs = horn_schunck(im1, im2, 1000, lmbd = 0.01, lk_init=True)
    # U_hs2, V_hs2 = horn_schunck(im1, im2, 1000, lmbd = 10, lk_init=True)
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(U_hs, V_hs, ax1_21, type='field',set_aspect=True)
    # show_flow(U_hs2, V_hs2, ax1_22, type='field',set_aspect=True)
    # fig1.suptitle(f'Horn-Schunck Optical Flow with different lambda')    
    # plt.show()
    
    
    
    
    # im1 = np.zeros((100,100), np.float32)
    # im1[30:70,30:70] = 1
    # im2 = np.roll(im1, shift=20, axis=1)
    
    # U_lk, V_lk = lucas_kanade(im1, im2, 3, harris = True)
    # U_lk2, V_lk2 = lucas_kanade(im1, im2, 3, harris = False)
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # # show_flow(U_lk, V_lk, ax1_21, type='angle')
    # # show_flow(U_lk2, V_lk2, ax1_22, type='angle')
    # magnitude = np.sqrt(U_lk**2 + V_lk**2)
    # ax1_21.imshow(magnitude, cmap='gray')
    # magnitude2 = np.sqrt(U_lk2**2 + V_lk2**2)
    # ax1_22.imshow(magnitude2, cmap='gray')
    # fig1.suptitle(f'Lucas-Kanade Optical Flow with and without Harris corner detection')    
    # plt.show()  
    
    # plot_flow_lk(im_translation1, im_translation2,harris=False)
    # plot_flow_lk(im_translation1, im_translation2,harris=True)
    
    # collision1 = cv2.imread('collision/00000140.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # collision2 = cv2.imread('collision/00000160.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_lk(collision1, collision2, window_size=9, harris=True)
    # plot_flow_pyramidal(collision1, collision2, N=9, num_levels=4, scale=0.5)
    
    # plot_flow_hs(collision1, collision2, lambda_=0.05, lk_init=True)
    # plot_flow_hs(collision1, collision2, lambda_=5.5, lk_init=True)
    # plot_flow_both_func(collision1, collision2,window_size=9, lambda_=2, num_iters=1000,harris=True, lk_init=True)
    


# DELETE ME LATER:
# im_translation1 = np.zeros((100,100), np.float32)
# im_translation1[30:70,30:70] = 1
# im_translation2 = np.roll(im_translation1, shift=10, axis=1)  # large motion
    
# im_translation1 = np.random.rand(100,100).astype(np.float32)
# im_translation2 = np.roll(im_translation1, shift=1, axis=1)




if __name__ == "__main__":
    
    
    images_for_second_part()
    
    # im1 = cv2.imread('disparity/office_left.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('disparity/office_right.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(im1,im2)

    # im1 = cv2.imread('collision/00000001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('collision/00000005.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # plot_flow_both_func(im1,im2)
    
    # im1 = cv2.imread('lab2/001.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # im2 = cv2.imread('lab2/004.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) #/ 255.0
    # # plot_flow_both_func(im1,im2)

    # plot_flow_pyramidal(im1, im2, N=3, num_levels=4, scale=0.5)

    # im1 = np.zeros((100,100), np.float32)
    # im1[30:70,30:70] = 1

    # im2 = np.roll(im1, shift=15, axis=1)  # large motion
    
    # u_lk, v_lk = lucas_kanade(im1, im2, 5)
    # u_pyr, v_pyr = pyramidal_lk(im1, im2, 5, num_levels=4, scale=0.5)
    
    
    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(u_lk, v_lk, ax1_21, type='angle')
    # show_flow(u_lk, v_lk, ax1_22, type='field', set_aspect=True)
    # fig1.suptitle('Lucas-Kanade Optical Flow')
    # plt.show()


    # fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2, cmap='gray')
    # show_flow(u_pyr, v_pyr, ax1_21, type='angle')
    # show_flow(u_pyr, v_pyr, ax1_22, type='field', set_aspect=True)
    # fig1.suptitle('Pyramidal Lucas-Kanade Optical Flow')
    # plt.show()

    # # time_comparison(im1, im2)
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

