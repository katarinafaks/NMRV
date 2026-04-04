import numpy as np
from matplotlib import pyplot as plt
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram,Tracker



# Implement mean shift algorithm. Estimate mean from data in the neighbourhood. Estimate
# the shift as the vector from the current mean to the estimated mean.


def get_neighbourhood(neighbourhood, center, size, kernel_shape = 'square'):
    """
    Creates mask of the kernel shape for the neighbourhood of the current position.
    Assume neighbourhood is a 2D array of size (H,W) with zeroes
    center is the center position (x,y) in the neighbourhood
    size is the size of the kernel 
    kernel_shape is the shape of the kernel, e.g. 'square' or 'circle'
    
    """
    
    H,W = neighbourhood.shape
    x,y = center
    
    for h in range(H):
        for w in range(W):
            if kernel_shape == 'square':
                if abs(h-y) <= size and abs(w-x) <= size:
                    neighbourhood[h,w] = 1
            if kernel_shape == 'circle':
                if (h-y)**2 + (w-x)**2 <= size**2:
                    neighbourhood[h,w] = 1
            else:
                raise ValueError("Unsupported kernel shape")
    return neighbourhood
        
    

def mean_shift_step(start_pos, h, data,sanity_check = False):
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
    
    # neighbourhood weights -- probability density?
    weights = data[y_min:y_max, x_min:x_max]
    
    #coordinate grid for the window np.meshgrid/np.mgrid?
    ys,xs = np.mgrid[y_min:y_max, x_min:x_max]

    # weighted mean:
    sum_w = np.sum(weights)
    
    if sum_w <= 0:
        return start_pos
    
    # algo looks at all points in the window, and calculates their center of gravity
    # based on how heavy (high value) each point is in the data. 
    
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
    


def mean_shift(start_pos, h, data, max_iters=100, tol=1e-3,sanity_check = False):
    save_x = [start_pos]
    
    x = start_pos
    for i in range(max_iters):
        new_x = mean_shift_step(x, h, data,sanity_check=sanity_check)
        shift_distance = np.linalg.norm(new_x - x)
        
        if shift_distance < tol:
            if sanity_check:
                print(f"Converged after {i+1} iterations with shift distance {shift_distance:.10f}.")
                plt.figure(figsize=(10, 5))
                plt.imshow(data)
                plt.scatter([pos[0] for pos in save_x[:-1]], [pos[1] for pos in save_x[:-1]], color='black', marker='x', label='Mean Shift Path')
                plt.scatter(save_x[-1][0], save_x[-1][1], color='red', marker='x', label='Final Position')
                plt.title(f'Mean Shift Convergence after {i+1} iterations')
                plt.legend()
                plt.show()
            break
        x = new_x
        save_x.append(x)
    return x
        


if __name__ == "__main__":
    temp = generate_responses_1()
    x = mean_shift(np.array([90, 80]), 3, temp, max_iters=100, tol=1e-5,sanity_check=True)
    



