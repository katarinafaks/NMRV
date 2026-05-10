import numpy as np
import cv2
from ex2_utils import Tracker, get_patch, extract_histogram, create_epanechnik_kernel
from ex4_utils import sample_gauss


class ParticleTrackerRW(Tracker):
    
    def name(self):
        return "ParticleTrackerRW"

    def initialize(self, image, region):
        self.num_particles = 100
        self.alpha = 0.05   # hist update factor
        self.sigma = 0.1    # sigma for weight to proba calc
        self.nbins = 16 
        
        self.q = 1      # noise factor for motion model
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        # force odd size za simetrijo s kernelom
        w = int(region[2])
        h = int(region[3])
        if w % 2 == 0: w += 1
        if h % 2 == 0: h += 1
        self.target_size = (w, h)
        

        
        self.center = np.array([region[0] + region[2]/2.0, region[1] + region[3]/2.0])
        
        # Using RW as motion model 
        self.A = np.eye(2, dtype=np.float32)
    
        
        q_val = self.q * np.linalg.norm(self.target_size)
        self.Q = q_val * np.array([[1, 0], [0, 1]], dtype=np.float32)
        
        # Init particles around the initial position with zero velocity and add gauss noise
        self.particles = np.zeros((self.num_particles, 2), dtype=np.float32)
        self.particles[:, 0] = self.center[0]
        self.particles[:, 1] = self.center[1]
        # Add noise to initial particles
        # self.particles += sample_gauss(np.zeros(4), self.Q, self.num_particles)
        self.particles = sample_gauss((self.center[0], self.center[1]), self.Q, self.num_particles)
        
        # init weights are uniform
        self.weights = np.ones(self.num_particles, dtype=np.float32)
        self.weights /= np.sum(self.weights)
        
        self.kernel = create_epanechnik_kernel(self.target_size[0], self.target_size[1], 
                                              np.linalg.norm(self.target_size)/2)
        
        # extract patch histogram as reference
        patch, _ = get_patch(image, self.center, self.target_size)
        self.target_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
        # normalize histogram:
        self.target_hist /= (np.sum(self.target_hist) + 1e-6)

    def track(self, image):
        # First resample particles according to weights
        
        weights = self.weights
        weights_cumsum = np.cumsum(weights)
        rand_samples = np.random.rand(self.num_particles)
        sampled_idx = np.digitize(rand_samples, weights_cumsum)
        sampled_particles = self.particles[sampled_idx.flatten(),:].reshape(-1, 2)
        
        s_gauss= sample_gauss(np.zeros(2), self.Q, self.num_particles)
        self.particles = np.dot(self.A, sampled_particles.T).T + s_gauss
        
        for i in range(self.num_particles):
            p_center = (self.particles[i, 0], self.particles[i, 1])
            patch, _ = get_patch(image, p_center, self.target_size)
            # if patch size is not correct => assign very low weight to this particle
            if patch.shape[0] != self.target_size[1] or patch.shape[1] != self.target_size[0]:
                self.weights[i] = 1e-10
                continue
            p_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
            p_hist /= (np.sum(p_hist) + 1e-6)
            # Hellinger distance: 
            dist = np.sqrt(max(0, 1.0 - np.sum(np.sqrt(p_hist * self.target_hist))))
            # convert to proba weight 
            self.weights[i] = np.exp(- (dist**2)/(2 *self.sigma**2))
        
        # normalize weights
        self.weights /= (np.sum(self.weights))
        
        # calc new position of target as weighted average of particles
        best_state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        self.center = (best_state[0], best_state[1])
        
        # update visual model (histogram of target)
        patch, _ = get_patch(image, self.center, self.target_size)
        if patch.shape[0] == self.target_size[1] and patch.shape[1] == self.target_size[0]:
            new_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
            new_hist /= (np.sum(new_hist) + 1e-6)
            self.target_hist = (1 - self.alpha) * self.target_hist + self.alpha * new_hist

        # return bounding box in format [x, y, w, h]
        return [self.center[0] - self.target_size[0]/2, self.center[1] - self.target_size[1]/2, 
                self.target_size[0], self.target_size[1],self.particles]
    
# Needed to run the tracker using run_tracker.py with ex2_utils.py
class ParticleParams():
    def __init__(self):
        pass