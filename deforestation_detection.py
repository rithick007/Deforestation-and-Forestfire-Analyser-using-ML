from deepforest.deforestation import DeforestationDetector
import os

def main():
    # Initialize the detector
    detector = DeforestationDetector(confidence_threshold=0.3)
    
    # Paths to your satellite images
    before_path = "path/to/before_image.tif"
    after_path = "path/to/after_image.tif"
    
    # Detect deforestation
    before_trees, after_trees, deforestation_prob = detector.detect_deforestation(
        before_path=before_path,
        after_path=after_path,
        patch_size=400,
        patch_overlap=0.05
    )
    
    # Print results
    print(f"Number of trees before: {len(before_trees)}")
    print(f"Number of trees after: {len(after_trees)}")
    print(f"Deforestation probability: {deforestation_prob:.2%}")
    
    # Visualize results
    output_dir = "deforestation_results"
    os.makedirs(output_dir, exist_ok=True)
    detector.visualize_deforestation(
        before_trees=before_trees,
        after_trees=after_trees,
        before_path=before_path,
        after_path=after_path,
        savedir=output_dir
    )

if __name__ == "__main__":
    main() 