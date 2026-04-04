import numpy as np
from matplotlib import pyplot as plt
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram,Tracker
from ex1_utils import gausssmooth



# Implement mean shift algorithm. Estimate mean from data in the neighbourhood. Estimate
# the shift as the vector from the current mean to the estimated mean.


def mean_shift_step(start_pos, h, data, kernel_shape='square', sanity_check = False):
    """    
    First implementation just for epanechnikov kernel, but should extend to other kernels as well.
    That's when the g should come in 
    
    g ... - gradient of the kernel function ? <- include?
    h ... kernel_size (bandwidth)
    data ... 2D array of data values (e.g. probability density)
    start_pos ... starting position for mean shift (2D coordinates)
    """
    
    xi,yi = start_pos[0], start_pos[1]    
    
    # window around the current position (xi, yi) with size h
    x_min = max(0, int(round(xi - h)))
    x_max = min(data.shape[1], int(round(xi + h + 1)))
    y_min = max(0, int(round(yi - h)))
    y_max = min(data.shape[0], int(round(yi + h + 1)))
    
    weights = np.copy(data[y_min:y_max, x_min:x_max] )
    ys,xs = np.mgrid[y_min:y_max, x_min:x_max]
    
    if kernel_shape == 'circle':
        dist_sq = (xs-xi)**2 + (ys-yi)**2
        mask = (dist_sq <= h**2).astype(np.float32) 
        weights *= mask
    elif kernel_shape == 'gaussian':
        dist_sq = (xs-xi)**2 + (ys-yi)**2
        kernel_weights = np.exp(-dist_sq / (2*h**2))
        weights *= kernel_weights
    elif kernel_shape == 'square':
        # no additional weighting needed for square kernel
        pass
    else:
        raise ValueError("Unsupported kernel shape")

    
    sum_w = np.sum(weights)
    if sum_w <= 0:
        return start_pos
    
    new_x = np.sum(xs*weights)/sum_w
    new_y = np.sum(ys*weights)/sum_w
    
    if sanity_check:
        plt.figure(figsize=(10, 5))
        plt.imshow(data)
        plt.scatter(xi, yi, color='red', marker='x', label='Initial Position')
        plt.scatter(new_x, new_y, color='blue', marker='x', label='New Position')
        plt.title('Mean Shift Step')
        plt.legend()
        plt.show()
    
    
    return np.array([new_x, new_y])
        
    

def mean_shift(start_pos, h, data, max_iters=100, tol=1e-3,sanity_check = False, kernel_shape='square', relax_factor = 1):
    save_x = [start_pos]
    
    
    x = start_pos
    for i in range(max_iters):
        new_x = mean_shift_step(x, h, data,kernel_shape=kernel_shape,sanity_check=False)
        shift_distance = np.linalg.norm(new_x - x)
        
        if shift_distance < tol:
            if sanity_check:
                save_x.append(new_x)
                # print(f"Converged after {i+1} iterations with shift distance {shift_distance:.10f}.")
                plt.figure(figsize=(10, 5))
                plt.imshow(data)
                plt.scatter([pos[0] for pos in save_x[:-1]], [pos[1] for pos in save_x[:-1]], color='black', marker='x', label='Mean Shift Path')
                plt.scatter(save_x[-1][0], save_x[-1][1], color='red', marker='x', label='Final Position')
                plt.title(f'Mean Shift Convergence after {i+1} iterations')
                plt.legend()
                plt.show()
            break
        
        if relax_factor > 1:            
            x_relaxed = x + relax_factor*(new_x - x)  # update position
            
            def get_val(pos):
                ix,iy = int(round(pos[0])), int(round(pos[1]))
                if 0 <= ix < data.shape[1] and 0 <= iy < data.shape[0]:
                    return data[iy, ix]
                else:
                    return -1
            if get_val(x_relaxed) > get_val(new_x):
                x = x_relaxed
            else:
                # we overshot
                x = new_x
                relax_factor = max(1.0, relax_factor * 0.8)  # reduce relax factor if we overshot <- cool down
        else:
            x = new_x        
        
        save_x.append(x)
    if sanity_check:
        return x,save_x
    return x
   
   
def generate_responses_2():
    """
    Generate 2D array with two distinct peaks.
    """
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[30, 30] = 1
    responses[70, 70] = 1 # a bit smaller but still distinct
    
    responses = gausssmooth(responses, 8) # smooth the responses to make them more realistic
    return responses

def generate_response_curve():
    responses = np.zeros((100, 100), dtype=np.float32)
    x_coords = np.linspace(20,80,100)
    y_coords = 45 + 0.02*(x_coords-50)**2
    
    for x,y in zip(x_coords, y_coords):
        if 0 <= x < 100 and 0 <= y < 100:
            responses[int(round(y)), int(round(x))] = 1
    
    return gausssmooth(responses, 5)
        
         
def for_report_plot(savex1,savex2,savex3,data):
    plt.figure(figsize=(5, 5))
    plt.imshow(data)
    plt.scatter([pos[0] for pos in savex1], [pos[1] for pos in savex1], color='black', marker='x')#, label='Mean Shift Path')
    plt.scatter(savex1[-1][0], savex1[-1][1], color='red', marker='x', label='Final Position')
    
    plt.scatter([pos[0] for pos in savex2], [pos[1] for pos in savex2], color='blue', marker='x')#, label='Mean Shift Path (h = 5)')
    plt.scatter(savex2[-1][0], savex2[-1][1], color='cyan', marker='x', label='Final Position')
    
    plt.scatter([pos[0] for pos in savex3], [pos[1] for pos in savex3], color='green', marker='x')#, label='Mean Shift Path (Gaussian Kernel)')
    plt.scatter(savex3[-1][0], savex3[-1][1], color='lime', marker='x', label='Final Position')
    
    plt.title('Mean Shift Convergence with Different starting points')
    plt.legend()
    plt.show()

def for_report_plot2(savex1,savex2,data):
    plt.figure(figsize=(5, 5))
    plt.imshow(data)
    iter1 = len(savex1)
    iter2 = len(savex2)
    plt.scatter([pos[0] for pos in savex1], [pos[1] for pos in savex1], color='black', marker='x', label='Mean-Shift Path (Gaussian) - iters: %d'%iter1)
    plt.scatter(savex1[-1][0], savex1[-1][1], color='red', marker='x')
    
    plt.scatter([pos[0] for pos in savex2], [pos[1] for pos in savex2], color='blue', marker='x', label='Mean-Shift Path (Epanechnikov) - iters: %d'%iter2)
    plt.scatter(savex2[-1][0], savex2[-1][1], color='cyan', marker='x')
    
    plt.title('Mean Shift Convergence with Different Kernels')
    plt.legend()
    plt.show()
    
def for_report_plot3(savex1,savex2,savex3,data):
    plt.figure(figsize=(5, 5))
    plt.imshow(data)
    
    plt.scatter([pos[0] for pos in savex3], [pos[1] for pos in savex3], color='black', marker='x', label='Mean Shift Path (kernel size = 10)')
    plt.scatter([pos[0] for pos in savex2], [pos[1] for pos in savex2], color='blue', marker='*', label='Mean Shift Path (kernel size = 7)')
    plt.scatter([pos[0] for pos in savex1], [pos[1] for pos in savex1], color='green', marker='o', label='Mean Shift Path (kernel size = 3)')
    
    plt.title('Mean Shift Convergence with Different kernel sizes')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    temp = generate_response_curve()
    print("Testing mean shift on a response curve...")
    x, save_x = mean_shift(np.array([54,70]), 5, temp, max_iters=100, tol=1e-7,sanity_check=True, kernel_shape='square')
    
    
    # temp = generate_responses_1()
    # print("Testing mean shift on with different relax factors...")
    # r_fact = [1,1.5,3,5,6.5,8]
    # for f in r_fact:
    #     x, save_x = mean_shift(np.array([45, 45]), 5, temp, 
    #                          max_iters=100, tol=1e-7,sanity_check=True, 
    #                          kernel_shape='square', relax_factor = f)
    #     mode = [50,70]
    #     dist = np.linalg.norm(x-mode)
    #     print(f"Relax factor: {f}, distance from true mode: {dist}, iterations: {len(save_x)}")
    
    # x2, save_x2 = mean_shift(np.array([80, 60]), 7, temp, max_iters=100, tol=1e-6,sanity_check=True, kernel_shape='square')
    # x3, save_x3 = mean_shift(np.array([60, 60]), 7, temp, max_iters=100, tol=1e-6,sanity_check=True, kernel_shape='square')

    # for_report_plot(save_x1, save_x2, save_x3, temp)
    # temp2 = generate_responses_2()
    # print("Testing mean shift on a response with two peaks...")
    # x1,save_x1 = mean_shift(np.array([45, 45]), 5, temp2, max_iters=100, tol=1e-5,sanity_check=True, kernel_shape='gaussian') 
    # x2,save_x2 = mean_shift(np.array([55, 55]), 5, temp2, max_iters=100, tol=1e-5,sanity_check=True, kernel_shape='square') 
    
    # for_report_plot2(save_x1, save_x2, temp2)
    
    # temp3 = generate_responses_1()
    # print("Testing mean shift with different kernels on a response with two peaks...")
    # x1,save_x1 = mean_shift(np.array([58, 45]), 3, temp3, max_iters=100, tol=1e-5,sanity_check=True, kernel_shape='square') 
    # x2,save_x2 = mean_shift(np.array([58, 45]), 7, temp3, max_iters=100, tol=1e-5,sanity_check=True, kernel_shape='square') 
    # x3,save_x3 = mean_shift(np.array([58, 45]), 10, temp3, max_iters=100, tol=1e-5,sanity_check=True, kernel_shape='square') 
    # for_report_plot3(save_x1, save_x2, save_x3, temp3)



