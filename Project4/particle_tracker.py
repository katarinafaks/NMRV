import numpy as np
import cv2
from ex2_utils import Tracker, get_patch, extract_histogram
from ex4_utils import sample_gauss

def create_epanechnik_kernel(width, height, sigma):
    # Ensure odd dimensions for a clear center
    w2 = width // 2
    h2 = height // 2
    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    
    # Calculate distance from center normalized by sigma
    dist = (X**2 + Y**2) / (sigma**2)
    kernel = 1 - dist
    kernel[kernel < 0] = 0
    return kernel / np.sum(kernel)

class ParticleTracker(Tracker):
    def name(self):
        return "ParticleTracker"

    def initialize(self, image, region):
        # 1. Parameter Setup
        self.num_particles = 100
        self.q = 0.05      # Process noise factor
        self.alpha = 0.05   # Model update rate
        self.sigma = 0.1    # Similarity sigma
        self.nbins = 16
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        # FIX: Force target size to be ODD to prevent kernel size mismatch
        w = int(region[2])
        h = int(region[3])
        if w % 2 == 0: w += 1
        if h % 2 == 0: h += 1
        self.target_size = (w, h)
        
        self.center = np.array([region[0] + region[2]/2.0, region[1] + region[3]/2.0])
        
        # 2. Motion Model (NCV)
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # q is set proportional to target size (diagonal)
        q_val = self.q * np.linalg.norm(self.target_size)
        self.Q = q_val * np.array([[1/3, 0, 1/2, 0], [0, 1/3, 0, 1/2], 
                                   [1/2, 0, 1, 0], [0, 1/2, 0, 1]], dtype=np.float32)
        
        # 3. Visual Model
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float32)
        self.particles[:, 0] = self.center[0]
        self.particles[:, 1] = self.center[1]
        self.weights = np.ones(self.num_particles, dtype=np.float32) / self.num_particles
        
        self.kernel = create_epanechnik_kernel(self.target_size[0], self.target_size[1], 
                                              np.linalg.norm(self.target_size)/2)
        
        patch, _ = get_patch(image, self.center, self.target_size)
        # Patch is now guaranteed to match kernel size
        self.target_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
        self.target_hist /= (np.sum(self.target_hist) + 1e-6)

    def track(self, image):
        # Resample
        weights_norm = self.weights / np.sum(self.weights)
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=weights_norm)
        self.particles = self.particles[indices]
        
        # Predict
        noise = sample_gauss(np.zeros(4), self.Q, self.num_particles)
        self.particles = np.dot(self.particles, self.A.T) + noise
        
        # Update Weights
        for i in range(self.num_particles):
            p_center = (self.particles[i, 0], self.particles[i, 1])
            patch, _ = get_patch(image, p_center, self.target_size)
            
            # Boundary check: if patch size is wrong, it's out of image
            if patch.shape[0] != self.target_size[1] or patch.shape[1] != self.target_size[0]:
                self.weights[i] = 1e-10
                continue
                
            p_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
            p_hist /= (np.sum(p_hist) + 1e-6)
            
            # Hellinger distance
            dist = np.sqrt(max(0, 1.0 - np.sum(np.sqrt(p_hist * self.target_hist))))
            self.weights[i] = np.exp(-0.5 * (dist**2) / (self.sigma**2))
            
        self.weights /= (np.sum(self.weights) + 1e-6)
        
        # State estimation (Weighted Mean)
        best_state = np.average(self.particles, axis=0, weights=self.weights)
        self.center = (best_state[0], best_state[1])
        
        # Template update
        patch, _ = get_patch(image, self.center, self.target_size)
        if patch.shape[0] == self.target_size[1] and patch.shape[1] == self.target_size[0]:
            new_hist = extract_histogram(patch, self.nbins, weights=self.kernel)
            new_hist /= (np.sum(new_hist) + 1e-6)
            self.target_hist = (1 - self.alpha) * self.target_hist + self.alpha * new_hist

        return [self.center[0] - self.target_size[0]/2, self.center[1] - self.target_size[1]/2, 
                self.target_size[0], self.target_size[1]]
    
# Needed to run the tracker using run_tracker.py with ex2_utils.py
class ParticleParams():
    def __init__(self):
        pass