# DeforestationAnalyser Algorithms

This document describes the algorithms used in the DeforestationAnalyser for detecting deforestation and forest fires from satellite imagery.

## 1. Deforestation Detection Algorithm

### Overview
The deforestation detection algorithm uses a combination of vegetation index analysis and image processing techniques to identify areas where forest cover has been removed between two temporal satellite images.

### Algorithm Steps

1. **Greenness Index Calculation**
   ```python
   # Calculate normalized green channel intensity
   green_index = green_channel / (mean(red_channel + blue_channel) + 0.1)
   ```
   - Uses the green channel dominance as a proxy for vegetation
   - Normalization against red and blue channels helps account for lighting variations
   - Small constant (0.1) prevents division by zero

2. **Change Detection**
   ```python
   greenness_decrease = max(0, before_green - after_green)
   ```
   - Computes the decrease in vegetation between images
   - Only positive changes are considered (areas that became less green)

3. **Adaptive Thresholding**
   ```python
   threshold = mean(greenness_decrease) + 1.0 * std(greenness_decrease)
   ```
   - Uses statistical properties of the image to determine significant changes
   - Adapts to different image conditions and seasons
   - Standard deviation scaling factor (1.0) can be adjusted for sensitivity

4. **Morphological Processing**
   - Erosion to remove noise
   - Dilation to connect nearby regions
   - Connected component analysis to identify distinct deforested areas
   - Size filtering to remove small, insignificant regions

### Metrics Calculation
- Total deforested area (m² and hectares)
- Number of distinct deforested regions
- Percentage of total land area affected
- Individual region statistics

## 2. Forest Fire Detection Algorithm

### Overview
The forest fire detection algorithm analyzes color and brightness changes to identify areas affected by forest fires, focusing on the characteristic signatures of burned vegetation.

### Algorithm Steps

1. **Brightness Analysis**
   ```python
   brightness = mean(RGB_channels)
   brightness_decrease = max(0, before_brightness - after_brightness)
   ```
   - Detects areas that became darker
   - Burned areas typically show decreased overall brightness

2. **Red Channel Dominance**
   ```python
   red_dominance = red_channel / (mean(green_channel + blue_channel) + 0.1)
   ```
   - Burned areas often show increased red channel intensity
   - Normalized against other channels to account for lighting variations

3. **Burn Index Calculation**
   ```python
   burn_index = brightness_decrease * red_dominance
   ```
   - Combines both factors for more accurate fire detection
   - Areas must show both darkening and increased red dominance

4. **Statistical Filtering**
   ```python
   threshold = mean(burn_index) + 2.0 * std(burn_index)
   ```
   - More aggressive thresholding to reduce false positives
   - Adapts to image characteristics automatically

### Region Processing
- Morphological operations to clean up detection
- Size-based filtering (minimum 150 pixels)
- Maximum burn area limitation (40% of total area)
- Connected component analysis for distinct burn regions

## Implementation Notes

1. **Image Preprocessing**
   - Handles multiple image formats (PNG, JPEG, TIFF)
   - Converts to consistent format (RGB)
   - Supports different image orientations (CHW, HWC)

2. **Performance Considerations**
   - Uses NumPy for efficient array operations
   - Implements memory-efficient processing for large images
   - Supports parallel processing where applicable

3. **Accuracy Factors**
   - Image quality and resolution
   - Seasonal variations
   - Atmospheric conditions
   - Time gap between images

## Limitations and Considerations

1. **Detection Accuracy**
   - Best results with clear, high-resolution images
   - May be affected by shadows and cloud cover
   - Seasonal changes can affect vegetation indices

2. **Environmental Factors**
   - Works best with similar lighting conditions between images
   - May need adjustment for different forest types
   - Weather conditions can affect detection accuracy

3. **Technical Requirements**
   - Minimum image resolution recommendations
   - Processing time increases with image size
   - Memory usage considerations for large images

## Future Improvements

1. **Potential Enhancements**
   - Machine learning-based classification
   - Additional vegetation indices (NDVI, EVI)
   - Atmospheric correction
   - Multi-spectral image support

2. **Planned Features**
   - Species-specific detection
   - Time series analysis
   - Cloud mask integration
   - Automated report generation 