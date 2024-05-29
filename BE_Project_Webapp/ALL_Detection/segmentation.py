import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from skimage.segmentation import mark_boundaries
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import cv2


class Segmentation:
    def __init__(self,image):
        self.image = image
        self.image_size = (256,256)
        self.kernel_size = (5, 5)
        self.image_segments = None

    
    def apply_gaussian_blur(self):
        # numpy_image = self.image.numpy()
        numpy_image = self.image
        blurred_image = cv2.GaussianBlur(numpy_image, self.kernel_size, 0)
        return blurred_image
    
    def build_segmentation_model(self,clusters):
        flat_image = self.image.flatten().reshape((-1, 1))
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        kmeans.fit(flat_image)
        labels = kmeans.labels_
        segmented_image = labels.reshape(self.image.shape)
        return segmented_image

    def separate_segmented_image(self,clusters,segmented_image):
        image_segments = []
        for i in range(clusters):
            segment_mask = (segmented_image == i)
            segment = np.zeros_like(self.image)
            segment[segment_mask] = self.image[segment_mask]
            image_segments.append(segment)
        
        self.image_segments = image_segments
    
    def compute_noise_metric(self,segment):
    # Compute image variance as the noise metric
        return np.var(segment)

    def find_segment_with_least_noise(self,segment_list):
        min_noise_segment = None
        min_noise_metric = float('inf')
        
        # Iterate over each segment in the list
        for idx, segment in enumerate(segment_list):
            # Compute noise metric for the current segment
            noise_metric = self.compute_noise_metric(segment)
            
            if noise_metric < min_noise_metric:
                min_noise_metric = noise_metric
                min_noise_segment = segment
        
        return min_noise_segment
    
    def extract_segment_with_least_noise(self):
        best_segment = self.find_segment_with_least_noise(self.image_segments)
        return best_segment

    def get_image_segment(self):
        blurred_image = self.apply_gaussian_blur()
        segmented_image = self.build_segmentation_model(6)
        self.separate_segmented_image(6,segmented_image)
        return self.extract_segment_with_least_noise()        
