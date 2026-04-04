import numpy as np
from matplotlib import pyplot as plt
from ex2_utils import generate_responses_1, extract_histogram, backproject_histogram,get_patch,create_epanechnik_kernel,Tracker
from ms_mode import mean_shift_step, mean_shift

#add gaussian kernel for mean-shift test
def create_gaussian_kernel(width, height, sigma):
    
    x = np.linspace(-(width - 1) / 2., (width - 1) / 2., width)
    y = np.linspace(-(height - 1) / 2., (height - 1) / 2., height)
    xx,yy = np.meshgrid(x, y)
    
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = kernel / np.max(kernel)
    return kernel
    


class MSParams():
    def __init__(self):
        self.kernel_type = 'epanechnikov' # 'epanechnikov', 'gaussian'
        self.sigma = 1 # for the kernel
        self.num_bins = 8
        self.eps = 1e-3 #epsilon for weighted histogram update (avoid division by zero)
        self.max_iters = 20
        self.tol = 0.5#5e-1 #0.5 <- adjust
        self.alpha = 0.5 # for histogram update
        # for kernel and patch:
        self.width = 21 
        self.height = 21


class MeanShiftTracker(Tracker):
    
    def initialize(self,image,region):
        """ initializes the tracker

        Args:
            image : current frame
            region : bounding box coordinates [x, y, width, height]
        """
        
        # from ncc_tracker
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        self.center = (region[0] + region[2]/2.0, region[1] + region[3]/2.0)
        self.window = region
        
        
        patch,_ = get_patch(image, self.center, (self.parameters.width, self.parameters.height))
        
        
        # create grid for mean-shift step 
        y_grid, x_grid = np.mgrid[0:self.parameters.height, 0:self.parameters.width]
        # center the grid around (0,0) -> (as in tutorial: -2,-1,0,1,2 for width=5)
        self.X = x_grid - int(self.parameters.width/2.0)
        self.Y = y_grid - int(self.parameters.height/2.0)
        
        if self.parameters.kernel_type == 'epanechnikov':
            self.kernel = create_epanechnik_kernel(self.parameters.width, self.parameters.height, self.parameters.sigma)
            self.g_kernel = np.ones((patch.shape[0], patch.shape[1])) # for epanechnikov kernel, g is constant within the kernel window
        elif self.parameters.kernel_type == 'gaussian':
            self.kernel = create_gaussian_kernel(self.parameters.width, self.parameters.height, self.parameters.sigma)
            self.g_kernel = self.kernel # for gaussian kernel, g is the same as the kernel
        else:
            raise ValueError("Unsupported kernel type")
        
        hist = extract_histogram(patch = patch, nbins = self.parameters.num_bins, weights=self.kernel)
        self.q = hist / np.sum(hist)
        

    def track(self,image):
        """
        Locates target by finding mode of the backprojection landscape?

        Args:
            image (2D RGB image): current video frame
        """

        # w_i = backprojection within extracted patch using weights v = \sqrt(q /(p + eps)) and then
        # new position using mean-shift equation. 
        # Update model using: q_new = (1-alpha)*q_old + alpha*q_, where q_ is the histogram using kernel


        x_upd = self.center[0]
        y_upd = self.center[1]
        
        w_win, h_win = self.parameters.width, self.parameters.height
        
        for i in range(self.parameters.max_iters):
            im_patch, _ = get_patch(image, (x_upd, y_upd), (w_win, h_win))
            
            # extract and normalize histogram of the current patch
            p = extract_histogram(patch = im_patch, nbins = self.parameters.num_bins, weights=self.kernel)
            p = p / np.sum(p + self.parameters.eps) # add eps to avoid division by zero
            
            # calculate weights q / (p + eps)
            v = np.sqrt(self.q / (p + self.parameters.eps))
            #v = v / np.sum(v) # normalize weights histogram ? 
        
            weight_map = backproject_histogram(patch = im_patch, histogram = v, nbins = self.parameters.num_bins)
        
            
            #apply g_kernel to weight map (if epanechnikov => no change, if gaussian => weighted by a gaussian)
            weight_map *= self.g_kernel
            
            w_sum = np.sum(weight_map)
            if w_sum <= 0: break
            
            change_x = np.sum(weight_map * self.X) / w_sum
            change_y = np.sum(weight_map * self.Y) / w_sum
            
            # if change in both directions small enough => stop
            if abs(change_x) < self.parameters.tol and abs(change_y) < self.parameters.tol:
                break
            
            x_upd += change_x
            y_upd += change_y
        
        
        # update window position (backk to top-left corner)<- we were tracking with center
        self.window[0] = x_upd - self.window[2]/2.0
        self.window[1] = y_upd - self.window[3]/2.0    
        self.center = (x_upd, y_upd)
        
        # update histogram model
        patch, _ = get_patch(image, self.center, (w_win, h_win))
        
        hist = extract_histogram(patch = patch, nbins = self.parameters.num_bins, weights=self.kernel)
        hist = hist / np.sum(hist)
        self.q =(1 - self.parameters.alpha)*self.q + self.parameters.alpha*hist # need to normalize histogram after update?
        self.q = self.q / np.sum(self.q) # <- needed or no???
        
        return self.window
            
        
    
        
            