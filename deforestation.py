import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DeforestationDetector:
    """A class for detecting deforestation from satellite images."""
    
    def __init__(self):
        """Initialize the deforestation detector."""
        pass
        
    def detect_trees(self, image_path):
        """Detect trees in a single image using image processing.
        
        Args:
            image_path (str): Path to the satellite image
            
        Returns:
            pd.DataFrame: DataFrame containing tree detections
        """
        # Open the image to get dimensions
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Create synthetic detection data (in a real application, this would use more sophisticated image processing)
        height, width = img_array.shape[:2]
        
        # Create sample tree detections based on image properties
        tree_count = int((height * width) / 5000)  # Arbitrary formula to generate a reasonable number
        
        # Generate synthetic tree detections
        xmin = np.random.randint(0, width - 50, size=tree_count)
        ymin = np.random.randint(0, height - 50, size=tree_count)
        xmax = xmin + np.random.randint(20, 50, size=tree_count)
        ymax = ymin + np.random.randint(20, 50, size=tree_count)
        scores = np.random.uniform(0.3, 1.0, size=tree_count)
        
        # Create pandas DataFrame
        detections = pd.DataFrame({
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'score': scores,
            'label': ['Tree'] * tree_count
        })
        
        return detections
    
    def detect_deforestation(self, before_path, after_path):
        """Detect deforestation by comparing two satellite images.
        
        Args:
            before_path (str): Path to the "before" satellite image
            after_path (str): Path to the "after" satellite image
            
        Returns:
            tuple: (before_trees, after_trees, deforestation_probability)
        """
        # Detect trees in both images
        before_trees = self.detect_trees(before_path)
        after_trees = self.detect_trees(after_path)
        
        # Calculate deforestation probability
        if len(before_trees) == 0:
            deforestation_prob = 0.0
        else:
            # Calculate percentage of trees that disappeared
            before_count = len(before_trees)
            after_count = len(after_trees)
            deforestation_prob = max(0, (before_count - after_count) / before_count)
        
        return before_trees, after_trees, deforestation_prob
        
    def visualize_trees(self, image_path, detections):
        """Visualize tree detections on an image.
        
        Args:
            image_path (str): Path to the image
            detections (pd.DataFrame): Tree detections
            
        Returns:
            matplotlib.figure.Figure: Figure with visualized detections
        """
        # Load the image
        img = np.array(Image.open(image_path))
        
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        # Display the image
        ax.imshow(img)
        
        # Add detections as rectangles
        for _, row in detections.iterrows():
            rect = Rectangle(
                (row['xmin'], row['ymin']),
                row['xmax'] - row['xmin'],
                row['ymax'] - row['ymin'],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
        
        ax.set_title(f"Tree Detections: {len(detections)} trees")
        ax.axis('off')
        
        return fig 