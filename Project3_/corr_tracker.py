"""
Implement simplified version of the MOSSE correlation filter.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'toolkit-dir', 'pytracking-toolkit-lite', 'utils'))
from tracker import Tracker  #To run toolkit without errors
# from ex2_utils import  Tracker #To run run_tracker.py without errors

from ex2_utils import  get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

import numpy as np
import cv2
from matplotlib import pyplot as plt





class CorrelationTracker(Tracker):


    def name(self):
        return "CorrelationTracker"
    
    def initialize(self, image, region):
        """
        Initialize Correlation Tracker
        """
        
        self.alpha = 0.1 #for filter update 0.1
        self.sigma = 2 # for gauss reponse
        self.lambda_ = 0.01 #for division by zero?
        self.large_region_factor = 1 #for initial target size - to capture more background for better filter learning
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        self.center = (region[0] + region[2]/2.0, region[1] + region[3]/2.0)
        self.width = region[2]
        self.height = region[3]
        self.target_size = (int(region[2]*self.large_region_factor),int(region[3]*self.large_region_factor))
        if self.target_size[0] % 2 == 0:
            self.target_size = (self.target_size[0]+1,self.target_size[1])
        if self.target_size[1] % 2 == 0:
            self.target_size = (self.target_size[0],self.target_size[1]+1)
        
        
        patch,_ = get_patch(image, self.center, self.target_size)
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        
        
        patch = patch - np.mean(patch)
        
        # create cosine window:
        self.cosine_window = create_cosine_window(self.target_size)
        f = patch * self.cosine_window
        # plt.subplot(1,2,1)
        # plt.imshow(f, cmap='gray')
        # plt.title('Cosine windowed patch')
        # plt.subplot(1,2,2)
        # plt.imshow(patch, cmap='gray')
        # plt.title('Initial patch')
        # plt.show()
        
        
        F_hat = np.fft.fft2(f)
        
        self.G = create_gauss_peak(self.target_size, self.sigma)
        self.G_hat = np.fft.fft2(self.G)
        # plt.imshow(self.G, cmap='gray')
        # plt.title('Gaussian peak')
        # plt.show()
        
        temp1 = self.G_hat * np.conj(F_hat)
        temp2 = F_hat * np.conj(F_hat) + self.lambda_
        self.H_hat_conj = temp1 / temp2
        # plt.imshow(np.log(np.abs(np.fft.fftshift(self.H_hat_conj)) + 1))
        # plt.title('Filter response')
        # plt.show()

    def track(self, image):
        """
        Track object
        """
        patch,_ = get_patch(image, self.center, self.target_size)
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        
        patch = patch - np.mean(patch) #needed?
        
        f = patch * self.cosine_window
        F_hat = np.fft.fft2(f)
        
        R = np.real(np.fft.ifft2(self.H_hat_conj * F_hat)) #np.abs? np.real? -> do we need just the real part here?
        # plt.imshow(R, cmap='gray')
        # plt.title('Correlation response')
        # plt.show()   
        
        # R_real = np.fft.fftshift(np.real(R)) # Shift so peak is in center for visualization
        # # Visualization for report:
        # plt.imshow(R_real, cmap='hot')
        # plt.title(f'Response map with sigma={self.sigma}')
        # plt.show()
        
        
        vmax,hmax = np.unravel_index(np.argmax(R), R.shape)
        #plot R and the peak:
        # plt.imshow(R, cmap='gray')
        # plt.scatter(hmax, vmax, color='red', marker='x')
        # plt.title('Correlation response with peak')
        # plt.show()
        

        # print(f"Raw Peak: {vmax, hmax}")
        #cetner peak relative to patch center -- period shift from fft?.could be a problem!
        if vmax > self.target_size[1] / 2:
            vmax -=  self.target_size[1]
        if hmax > self.target_size[0] / 2:
            hmax -= self.target_size[0]
        # print(f"Shifted Peak: {vmax, hmax}")
        self.center = (self.center[0] + hmax, self.center[1] + vmax)
        
        # plot the image with the new and old center:
        # plt.imshow(image, cmap='gray')
        # plt.scatter(self.center[0], self.center[1], color='red', marker='x', label='New center')
        # plt.scatter(self.center[0] - hmax, self.center[1] - vmax, color='blue', marker='x', label='Old center')
        # plt.legend()
        # plt.title('Image with new and old center')
        # plt.show()
        
        #update the filter:
        new_patch,_ = get_patch(image, self.center, self.target_size)
        if len(new_patch.shape) == 3:
            new_patch = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
        new_patch = new_patch.astype(np.float32)
        
        new_patch = new_patch - np.mean(new_patch)
        

        
        f_new = new_patch * self.cosine_window
        F_hat_new = np.fft.fft2(f_new)
        F_hat_new_conj = np.conj(F_hat_new)
        
        temp1 = self.G_hat * F_hat_new_conj
        temp2 = F_hat_new * F_hat_new_conj + self.lambda_
        H_hat_conj_new = temp1 / temp2 
        
        self.H_hat_conj = (1 - self.alpha) * self.H_hat_conj + self.alpha * H_hat_conj_new
        
        
        # print(f"Max filter response: {np.max(self.H_hat_conj)}")
        
        #return the new bounding box:
        bb = [self.center[0] - self.width/2, 
              self.center[1] - self.height/2, 
               self.width, self.height] 
        return bb
        
        

# Needed to run the tracker using run_tracker.py with ex2_utils.py
class CorrelationParams():
    def __init__(self):
        pass
