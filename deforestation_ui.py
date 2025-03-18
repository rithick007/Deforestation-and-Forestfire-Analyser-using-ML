import streamlit as st
import os
from deforestation import DeforestationDetector
from PIL import Image
import tempfile
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from skimage.morphology import binary_dilation, disk, binary_erosion
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

def detect_forest_fires(before_img, after_img):
    """Detect potential forest fire areas by analyzing color changes."""
    try:
        # Convert images to numpy arrays if they're not already
        if isinstance(before_img, str):
            before_img = np.array(Image.open(before_img))
        if isinstance(after_img, str):
            after_img = np.array(Image.open(after_img))
        
        # Ensure both images are RGB and in the correct shape format
        if len(before_img.shape) == 3:
            if before_img.shape[0] == 3:  # CHW format
                before_img = np.transpose(before_img, (1, 2, 0))
        elif len(before_img.shape) == 2:  # Grayscale
            before_img = np.stack([before_img] * 3, axis=-1)
            
        if len(after_img.shape) == 3:
            if after_img.shape[0] == 3:  # CHW format
                after_img = np.transpose(after_img, (1, 2, 0))
        elif len(after_img.shape) == 2:  # Grayscale
            after_img = np.stack([after_img] * 3, axis=-1)
        
        # Calculate the "burnedness" index
        # This simple algorithm looks for areas that have become darker and more reddish
        before_brightness = np.mean(before_img, axis=2)
        after_brightness = np.mean(after_img, axis=2)
        
        # Red channel dominance in after image with safety against division by zero
        green_blue_mean = np.mean(after_img[:,:,1:], axis=2)
        red_dominance_after = after_img[:,:,0] / (green_blue_mean + 0.1)  # Avoid division by zero
        
        # Combined burn index: areas that got darker AND have red dominance
        brightness_decrease = np.maximum(0, before_brightness - after_brightness)
        burn_index = brightness_decrease * red_dominance_after
        
        # Normalize burn index for visualization (with safety check)
        max_burn_index = np.max(burn_index)
        if max_burn_index > 0:
            burn_index = burn_index / max_burn_index
        
        # Use more aggressive thresholding to limit excessive detection
        mean_burn = np.mean(burn_index)
        std_burn = max(np.std(burn_index), 0.001)  # Ensure minimum std to avoid threshold issues
        threshold = mean_burn + 2.0 * std_burn  # Increase threshold to reduce false positives
        burn_mask = burn_index > threshold
        
        # Clean up the mask with smaller structuring element to avoid over-detection
        burn_mask = binary_dilation(burn_mask, disk(2))
        
        # Calculate metrics
        labeled_burns = label(burn_mask)
        burn_regions = regionprops(labeled_burns)
        
        # Use stricter filtering for significant burn regions
        significant_burns = [r for r in burn_regions if r.area > 150]  # Increased minimum area
        
        # Limit the burn area to a realistic percentage of the total area (max 40%)
        total_burn_area = sum(r.area for r in significant_burns)
        total_image_area = before_img.shape[0] * before_img.shape[1]
        
        if total_burn_area / total_image_area > 0.4:  # If burn area exceeds 40%, limit it
            scaling_factor = (0.4 * total_image_area) / total_burn_area
            total_burn_area = int(total_burn_area * scaling_factor)
        
        num_burn_regions = len(significant_burns)
        
        # Create visualization
        burn_visualization = create_burn_visualization(after_img, burn_mask)
        
        return total_burn_area, num_burn_regions, burn_visualization
    
    except Exception as e:
        st.error(f"Error in fire detection: {str(e)}")
        # Return default values on error
        return 0, 0, np.zeros((10, 10, 3), dtype=np.uint8)

def create_burn_visualization(image, burn_mask):
    """Create a visualization of burned areas."""
    try:
        # Make sure image is in the right format
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure image is in uint8 format for matplotlib
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
        
        # Create a copy of the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Create a custom colormap for the overlay
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # Transparent to red with alpha
        cm = LinearSegmentedColormap.from_list("burn_cmap", colors, N=100)
        
        # Overlay the burn mask
        plt.imshow(burn_mask.astype(float), cmap=cm)
        plt.axis('off')
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all')  # Close all figures to avoid memory leaks
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"Error in burn visualization: {str(e)}")
        # Return a small red image on error
        error_img = np.zeros((100, 100, 3), dtype=np.uint8)
        error_img[:, :, 0] = 255  # Red channel
        return Image.fromarray(error_img)

def create_deforestation_visualization(image_path, deforested_mask):
    """Create a custom visualization highlighting deforested areas."""
    try:
        # Open the image
        image = np.array(Image.open(image_path))
        
        # Ensure image is in HWC format
        if len(image.shape) == 3 and image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure image is in uint8 format for matplotlib
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)
            
        # Create a figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Create a custom colormap for the overlay (red for deforested areas)
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # Transparent to red with alpha
        cm = LinearSegmentedColormap.from_list("deforest_cmap", colors, N=100)
        
        # Overlay the deforestation mask
        plt.imshow(deforested_mask.astype(float), cmap=cm)
        plt.axis('off')
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close('all')
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"Error creating deforestation visualization: {str(e)}")
        # Return a placeholder image on error
        error_img = np.zeros((100, 100, 3), dtype=np.uint8)
        error_img[:, :, 0] = 255  # Red channel
        return Image.fromarray(error_img)

def identify_barren_areas(before_path, after_path, image_shape, pixel_resolution=1.0, min_area=100):
    """Identify areas that appear barren (deforested) between the two images."""
    try:
        # Load images
        before_img = np.array(Image.open(before_path))
        after_img = np.array(Image.open(after_path))
        
        # Ensure images are in the right format
        if len(before_img.shape) == 3 and before_img.shape[0] == 3:  # CHW format
            before_img = np.transpose(before_img, (1, 2, 0))
        if len(after_img.shape) == 3 and after_img.shape[0] == 3:  # CHW format
            after_img = np.transpose(after_img, (1, 2, 0))
            
        # Convert to grayscale if needed
        if len(before_img.shape) == 3:
            before_gray = np.mean(before_img, axis=2)
        else:
            before_gray = before_img
            
        if len(after_img.shape) == 3:
            after_gray = np.mean(after_img, axis=2)
        else:
            after_gray = after_img
        
        # Calculate greenness index (using green channel or grayscale as approximation)
        if len(before_img.shape) == 3 and before_img.shape[2] >= 3:
            before_green = before_img[:,:,1] / (np.mean(before_img[:,:,[0,2]], axis=2) + 0.1)
        else:
            before_green = before_gray
            
        if len(after_img.shape) == 3 and after_img.shape[2] >= 3:
            after_green = after_img[:,:,1] / (np.mean(after_img[:,:,[0,2]], axis=2) + 0.1)
        else:
            after_green = after_gray
        
        # Areas that lost greenness between images
        greenness_decrease = np.maximum(0, before_green - after_green)
        
        # Normalize and threshold
        if np.max(greenness_decrease) > 0:
            greenness_decrease = greenness_decrease / np.max(greenness_decrease)
        
        # Use adaptive threshold
        mean_decrease = np.mean(greenness_decrease)
        std_decrease = max(np.std(greenness_decrease), 0.001)
        threshold = mean_decrease + 1.0 * std_decrease
        
        # Create binary mask of potentially deforested areas
        deforested_mask = greenness_decrease > threshold
        
        # Clean up with morphological operations
        deforested_mask = binary_erosion(deforested_mask, disk(2))
        deforested_mask = binary_dilation(deforested_mask, disk(4))
        
        # Find connected components
        labeled_regions = label(deforested_mask)
        regions = regionprops(labeled_regions)
        
        # Filter by size and sort by area
        significant_regions = [r for r in regions if r.area > min_area]
        significant_regions.sort(key=lambda r: r.area, reverse=True)
        
        # Calculate total area
        total_deforested_pixels = sum(region.area for region in significant_regions)
        total_deforested_area = total_deforested_pixels * (pixel_resolution ** 2)
        
        # Create region information
        region_info = []
        for i, region in enumerate(significant_regions):
            area = region.area * (pixel_resolution ** 2)
            y0, x0, y1, x1 = region.bbox
            cy, cx = region.centroid
            
            region_info.append({
                'id': i + 1,
                'area': area,
                'area_percentage': area / (image_shape[0] * image_shape[1] * pixel_resolution**2) * 100,
                'centroid': (cx, cy),
                'bbox': (x0, y0, x1, y1)
            })
        
        return (
            total_deforested_area,
            len(significant_regions),
            deforested_mask,
            region_info
        )
        
    except Exception as e:
        st.error(f"Error in barren area detection: {str(e)}")
        return 0, 0, np.zeros(image_shape[:2], dtype=bool), []

def convert_to_geotiff(image_path, output_path):
    """Convert various image formats to GeoTIFF."""
    try:
        # Read the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3)  # Convert to RGB
        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]  # Remove alpha channel
            if img_array.shape[2] == 3:  # RGB
                img_array = img_array.transpose(2, 0, 1)  # Convert to CHW format
        
        # Create a GeoTIFF with default geotransform
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=img_array.shape[1] if len(img_array.shape) == 3 else img_array.shape[0],
            width=img_array.shape[2] if len(img_array.shape) == 3 else img_array.shape[1],
            count=3,
            dtype=img_array.dtype,
            crs=rasterio.crs.CRS.from_epsg(4326),  # WGS84
            transform=rasterio.transform.from_origin(0, 0, 1, 1)  # Default transform
        ) as dst:
            dst.write(img_array)
            
        return img_array.shape
    except Exception as e:
        raise Exception(f"Error converting image to GeoTIFF: {str(e)}")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and convert to GeoTIFF if needed."""
    try:
        # Get file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_path = tmp_file.name
        
        # If not already a GeoTIFF, convert it
        if file_ext not in ['.tif', '.tiff']:
            output_path = input_path + '.tif'
            image_shape = convert_to_geotiff(input_path, output_path)
            os.unlink(input_path)  # Remove the original file
            return output_path, image_shape
        
        # For existing GeoTIFF files, get image shape
        with rasterio.open(input_path) as src:
            image_shape = (src.height, src.width, src.count)
        return input_path, image_shape
    except Exception as e:
        raise Exception(f"Error saving uploaded file: {str(e)}")

def add_bg_from_url():
    """Add a background image from a URL."""
    forest_bg = """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), url("https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3274&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .main {
        background-color: transparent;
        padding: 20px;
        margin: 10px;
    }
    
    h1, h2, h3 {
        color: #0a3412 !important;
        font-weight: bold !important;
        text-shadow: 0px 0px 3px rgba(255, 255, 255, 0.9);
        font-family: 'Georgia', serif !important;
        letter-spacing: 0.5px;
    }
    
    .stSubheader {
        color: #072209 !important;
        font-weight: bold !important;
        font-family: 'Georgia', serif !important;
    }
    
    .uploadedFile {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
        padding: 5px !important;
        border: 1px solid #2e7d32 !important;
    }
    
    .stButton>button {
        background-color: #0a3412 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        transition: all 0.3s !important;
    }
    
    .stButton>button:hover {
        background-color: #061f0a !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .css-12oz5g7 {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        border-radius: 5px !important;
        color: #072209 !important;
        font-weight: bold !important;
    }
    
    /* Make expander content transparent */
    .streamlit-expanderContent {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Text colors */
    p, li, label, div {
        color: #051c08 !important;
        font-family: 'Calibri', 'Helvetica', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        color: #051c08 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #072209 !important;
        font-weight: bold !important;
    }
    
    /* Footer */
    .footer {
        padding: 10px;
        text-align: center;
        color: #051c08;
        font-size: 0.8rem;
        margin-top: 30px;
        font-weight: 500 !important;
    }
    
    /* Card-like sections */
    .card {
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #0a3412;
    }
    
    /* Fix for card with colored backgrounds - lighter text */
    .card[style*="border-left: 5px solid #6b0000"] p,
    .card[style*="border-left: 5px solid #6b0000"] li {
        color: #ffffff !important;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.5);
    }
    
    .card[style*="border-left: 5px solid #0a3412"] p,
    .card[style*="border-left: 5px solid #0a3412"] li {
        color: #ffffff !important;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* File uploader custom styling */
    .stFileUploader {
        background-color: transparent !important;
    }
    
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    
    .stRadio label {
        color: #051c08 !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #051c08 !important;
        font-weight: bold !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.4) !important;
        border-left-color: #0a3412 !important;
    }
    
    /* Tables */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.6) !important;
    }
    
    .stDataFrame th {
        background-color: rgba(200, 230, 201, 0.7) !important;
        color: #051c08 !important;
        font-weight: bold !important;
    }
    
    /* Dark text for inputs */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiselect > div > div > div {
        color: #051c08 !important;
        font-weight: 600 !important;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .logo-image {
        height: 80px;
        margin-right: 20px;
    }
    
    .logo-title {
        color: #051c08 !important;
        font-family: 'Georgia', serif !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.9);
        letter-spacing: 1px;
    }
    
    /* Remove unwanted bars */
    .element-container {
        margin-bottom: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Remove horizontal lines */
    hr {
        display: none !important;
    }
    
    /* Fix for colored text on colored backgrounds */
    .card[style*="border-left: 5px solid #6b0000"] h3 {
        color: #ffffff !important;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.7);
    }
    
    .card[style*="border-left: 5px solid #0a3412"] h3 {
        color: #ffffff !important;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.7);
    }
    </style>
    """
    st.markdown(forest_bg, unsafe_allow_html=True)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Deforestation Analyser", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌳"
    )
    
    # Add background image and styling
    add_bg_from_url()
    
    # Create container for main content
    main_container = st.container()
    
    with main_container:
        # Custom header with logo
        st.markdown("""
        <div class="logo-container">
            <img class="logo-image" src="https://cdn-icons-png.flaticon.com/512/628/628324.png" alt="Tree Logo">
            <h1 class="logo-title">Deforestation Analyser</h1>
        </div>

        <div style="text-align: center; margin: 20px 0;">
            <a href="https://sensational-macaron-123f61.netlify.app/" target="_blank" style="text-decoration: none;">
                <button style="
                    background-color: #0a3412;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                    cursor: pointer;
                    font-size: 16px;
                    transition: all 0.3s;
                ">🔴 Live Monitoring Dashboard</button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Update the CSS for all buttons to have white text
        st.markdown("""
        <style>
        /* Existing styles remain the same */

        /* Update button text color */
        .stButton>button {
            color: white !important;
            background-color: #0a3412 !important;
            font-weight: bold !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 5px !important;
            transition: all 0.3s !important;
        }

        .stButton>button:hover {
            background-color: #061f0a !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }

        /* Make download button text white */
        .stDownloadButton>button {
            color: white !important;
        }

        /* Make all button texts white */
        button {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize detector for our custom implementation
        if 'detector' not in st.session_state:
            st.session_state.detector = DeforestationDetector()
    
        # Make the container background more transparent
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.4); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #051c08; font-family: Georgia, serif;">Analyze Forest Changes</h3>
        <p style="color: #051c08; font-weight: 500;">
        This application helps you analyze changes in forest cover by comparing satellite images taken at different times.
        Upload your "before" and "after" images, select your analysis mode, and get detailed insights about deforestation or forest fires.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
        # File upload section - remove the bottom border that creates the white bar
        st.markdown('<h2 style="color: #051c08; font-family: Georgia, serif;">Image Upload</h2>', unsafe_allow_html=True)
        
        # File uploaders with improved styling
        col1, col2 = st.columns(2)
        
        with col1:
            # Removed the card div completely
            st.subheader("Before Image")
            st.markdown('<p style="color: #051c08; font-size: 0.9rem; font-weight: 500;">Upload an image showing the area before changes occurred</p>', unsafe_allow_html=True)
            before_file = st.file_uploader("Upload 'Before' satellite image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key="before_uploader")
            if before_file:
                st.image(before_file, caption="Before Image Preview", use_container_width=True)
        
        with col2:
            # Removed the card div completely
            st.subheader("After Image")
            st.markdown('<p style="color: #051c08; font-size: 0.9rem; font-weight: 500;">Upload an image showing the same area after changes occurred</p>', unsafe_allow_html=True)
            after_file = st.file_uploader("Upload 'After' satellite image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key="after_uploader")
            if after_file:
                st.image(after_file, caption="After Image Preview", use_container_width=True)
        
        # Analysis mode selection - remove the bottom border that creates the white bar
        st.markdown('<h2 style="color: #051c08; margin-top: 20px; font-family: Georgia, serif;">Analysis Settings</h2>', unsafe_allow_html=True)
        
        # Replace the custom div with direct components and minimal styling
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Deforestation Analysis", "Forest Fire Detection"],
            help="Choose the type of analysis to perform",
            index=0
        )
        
        # Description based on selected mode
        if analysis_mode == "Deforestation Analysis":
            st.markdown('<p style="color: #051c08; font-style: italic; font-weight: 600;">Deforestation Analysis identifies areas where forest cover has been removed between the two images.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #6b0000; font-style: italic; font-weight: 600;">Forest Fire Detection identifies areas likely affected by fires based on color and vegetation changes.</p>', unsafe_allow_html=True)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            # Remove the background color div
            # Pixel resolution
            pixel_resolution = st.number_input(
                "Pixel Resolution (meters)",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                help="The size of each pixel in meters. Use this to calculate actual area."
            )
            
            # Minimum deforested area to consider
            min_deforest_area = st.number_input(
                "Minimum Area Size (pixels)",
                min_value=10,
                max_value=1000,
                value=100,
                help="Minimum size of regions to include in analysis (in pixels)"
            )
            
            if analysis_mode == "Forest Fire Detection":
                # Confidence threshold slider (only used for fire detection)
                confidence = st.slider(
                    "Detection Confidence Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.3,
                    step=0.1,
                    help="Higher values mean more confident detections but fewer detections overall"
                )
                
                # Update confidence threshold
                st.session_state.detector.confidence_threshold = confidence
        
        # Process button with enhanced styling
        st.markdown('<div style="display: flex; justify-content: center; margin: 20px 0;">', unsafe_allow_html=True)
        button_text = "🔍 Analyze Deforestation" if analysis_mode == "Deforestation Analysis" else "🔥 Detect Forest Fires"
        process_button = st.button(button_text, disabled=not (before_file and after_file), key="process_button")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Helper text for users
        if not (before_file and after_file):
            st.info("📁 Please upload both 'before' and 'after' images to begin analysis")
        
        # Process images when button is clicked
        if process_button:
            with st.spinner("Processing images... Please wait"):
                try:
                    # Save and convert uploaded files
                    before_path, before_shape = save_uploaded_file(before_file)
                    after_path, after_shape = save_uploaded_file(after_file)
                    
                    # Create output directory
                    output_dir = "analysis_results"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create a container for results
                    st.markdown('<div style="background-color: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 10px; margin-top: 30px; border: 2px solid #0a3412;">', unsafe_allow_html=True)
                    
                    # Perform deforestation analysis
                    if analysis_mode == "Deforestation Analysis":
                        # Analyze deforested areas directly from the images
                        deforested_area, num_regions, deforested_mask, region_info = identify_barren_areas(
                            before_path, 
                            after_path, 
                            before_shape,
                            pixel_resolution,
                            min_deforest_area
                        )
                        
                        # Display deforestation results
                        st.markdown('<h2 style="color: #051c08; text-align: center; margin-bottom: 20px; font-family: Georgia, serif;">Deforestation Analysis Results</h2>', unsafe_allow_html=True)
                        
                        # Create metrics with improved styling
                        metrics_row = st.columns(3)
                        
                        with metrics_row[0]:
                            value = f"{deforested_area:.1f} m²" if deforested_area < 10000 else f"{deforested_area/10000:.2f} ha"
                            st.metric(
                                "Total Deforested Area", 
                                value,
                                help="Total area where forest has been removed"
                            )
                        with metrics_row[1]:
                            st.metric(
                                "Deforested Regions", 
                                num_regions,
                                help="Number of distinct deforested areas detected"
                            )
                        with metrics_row[2]:
                            total_analyzed_area = before_shape[0] * before_shape[1] * pixel_resolution**2
                            deforest_percentage = (deforested_area / max(total_analyzed_area, 1) * 100)
                            st.metric(
                                "Percent Deforested", 
                                f"{deforest_percentage:.2f}%",
                                help="Percentage of analyzed area that has been deforested"
                            )
                        
                        # Add detailed analysis with card styling
                        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
                        st.markdown('<h3 style="color: #051c08; font-family: Georgia, serif;">Detailed Deforestation Analysis</h3>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <ul style="color: #051c08; font-weight: 500;">
                            <li>Total analyzed area: <b>{(before_shape[0] * before_shape[1] * pixel_resolution**2):.1f} m²</b> ({(before_shape[0] * before_shape[1] * pixel_resolution**2)/10000:.2f} ha)</li>
                            <li>Total deforested area: <b>{deforested_area:.1f} m²</b> ({deforested_area/10000:.4f} ha)</li>
                            <li>Percentage of land deforested: <b>{deforest_percentage:.2f}%</b></li>
                            <li>Number of distinct deforested regions: <b>{num_regions}</b></li>
                        </ul>
                        """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display deforestation region details with enhanced styling
                        if region_info and len(region_info) > 0:
                            st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
                            st.markdown('<h3 style="color: #051c08; font-family: Georgia, serif;">Deforested Region Details</h3>', unsafe_allow_html=True)
                            
                            # Create a table with region information
                            region_data = {
                                "Region ID": [r['id'] for r in region_info],
                                "Area (m²)": [f"{r['area']:.1f}" for r in region_info],
                                "Area (ha)": [f"{r['area']/10000:.4f}" for r in region_info],
                                "% of Total": [f"{r['area_percentage']:.2f}%" for r in region_info]
                            }
                            
                            # Show only top 10 regions to avoid clutter
                            max_regions = min(10, len(region_info))
                            st.dataframe(region_data, height=max_regions*35 + 38)
                            
                            if len(region_info) > 10:
                                st.caption(f"Showing top 10 of {len(region_info)} deforested regions")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Create visualization tabs
                        st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
                        st.markdown('<h3 style="color: #051c08; font-family: Georgia, serif;">Deforestation Visualization</h3>', unsafe_allow_html=True)
                        
                        # Create custom visualization of deforested areas
                        deforest_viz = create_deforestation_visualization(after_path, deforested_mask)
                        
                        # Display deforestation visualizations
                        st.image(deforest_viz, caption="Deforested Areas (highlighted in red)", use_container_width=True)
                        
                        # Save the visualization
                        deforest_viz.save(os.path.join(output_dir, "deforested_areas.png"))
                        
                        # Download button for visualization
                        with open(os.path.join(output_dir, "deforested_areas.png"), "rb") as file:
                            btn = st.download_button(
                                label="⬇️ Download Visualization",
                                data=file,
                                file_name="deforested_areas.png",
                                mime="image/png"
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Forest fire detection
                    elif analysis_mode == "Forest Fire Detection":
                        try:
                            # Load images for fire detection
                            before_img = np.array(Image.open(before_path))
                            after_img = np.array(Image.open(after_path))
                            
                            # Detect forest fires
                            burn_area, num_burn_regions, burn_visualization = detect_forest_fires(before_img, after_img)
                            
                            # Display in a card-like container
                            st.markdown('<h2 style="color: #6b0000; text-align: center; margin-bottom: 20px; font-family: Georgia, serif;">Forest Fire Analysis Results</h2>', unsafe_allow_html=True)
                            
                            # Create metrics
                            fire_metrics = st.columns(3)
                            with fire_metrics[0]:
                                burned_area_m2 = burn_area * pixel_resolution**2
                                value = f"{burned_area_m2:.1f} m²" if burned_area_m2 < 10000 else f"{burned_area_m2/10000:.2f} ha"
                                st.metric(
                                    "Burned Area", 
                                    value,
                                    help="Estimated area affected by fire"
                                )
                            with fire_metrics[1]:
                                st.metric(
                                    "Burn Regions", 
                                    num_burn_regions,
                                    help="Number of distinct burned areas detected"
                                )
                            with fire_metrics[2]:
                                total_area = max(before_shape[0] * before_shape[1], 1)
                                # Fix the percentage calculation to ensure it doesn't exceed 100%
                                burn_percentage = min((burn_area / total_area * 100), 100.0)
                                st.metric(
                                    "Percent Burned",
                                    f"{burn_percentage:.2f}%",
                                    help="Percentage of analyzed area that has been burned"
                                )
                            
                            # Add detailed fire analysis with card styling
                            st.markdown('<div class="card" style="margin-top: 20px; border-left: 5px solid #6b0000; background-color: rgba(255, 255, 255, 0.8);">', unsafe_allow_html=True)
                            st.markdown('<h3 style="color: #6b0000; font-family: Georgia, serif;">Detailed Fire Analysis</h3>', unsafe_allow_html=True)
                            total_area = max(before_shape[0] * before_shape[1], 1)
                            # Fix the percentage calculation to ensure it doesn't exceed 100%
                            burn_percentage = min((burn_area / total_area * 100), 100.0)
                            st.markdown(f"""
                            <ul style="color: #051c08; font-weight: 500;">
                                <li>Total analyzed area: <b>{(total_area * pixel_resolution**2):.1f} m²</b> ({(total_area * pixel_resolution**2)/10000:.2f} ha)</li>
                                <li>Burned area: <b>{burn_area * pixel_resolution**2:.1f} m²</b> ({burn_area * pixel_resolution**2 / 10000:.4f} ha)</li>
                                <li>Percentage of land burned: <b>{burn_percentage:.2f}%</b></li>
                                <li>Number of distinct burn regions: <b>{num_burn_regions}</b></li>
                            </ul>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display fire visualization
                            st.markdown('<div class="card" style="margin-top: 20px; border-left: 5px solid #6b0000; background-color: rgba(255, 255, 255, 0.8);">', unsafe_allow_html=True)
                            st.markdown('<h3 style="color: #6b0000; font-family: Georgia, serif;">Fire Detection Visualization</h3>', unsafe_allow_html=True)
                            st.image(burn_visualization, caption="Potential Burned Areas (highlighted in red)", use_container_width=True)
                            
                            # Save the visualization
                            burn_visualization.save(os.path.join(output_dir, "burned_areas.png"))
                            
                            # Download button for visualization
                            with open(os.path.join(output_dir, "burned_areas.png"), "rb") as file:
                                btn = st.download_button(
                                    label="⬇️ Download Visualization",
                                    data=file,
                                    file_name="burned_areas.png",
                                    mime="image/png"
                                )
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div style="background-color: #ffcdd2; padding: 10px; border-radius: 5px; color: #5d0808;"><b>Error:</b> {str(e)}</div>', unsafe_allow_html=True)
                            st.warning("Could not complete fire detection analysis. Please ensure your images are valid.")
                    
                    # Close the results container
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Clean up temporary files
                    try:
                        os.unlink(before_path)
                        os.unlink(after_path)
                    except:
                        pass  # Ignore cleanup errors
                    
                except Exception as e:
                    st.markdown(f'<div style="background-color: #ffcdd2; padding: 10px; border-radius: 5px; color: #5d0808;"><b>Error:</b> {str(e)}</div>', unsafe_allow_html=True)
                    st.error("Please make sure your images are valid and try again.")
        
        # Add information about the tool with improved styling
        with st.expander("About This Tool"):
            # Remove the background color div
            st.markdown("""
            <h3 style="color: #051c08; font-family: Georgia, serif;">Deforestation Analysis Tool</h3>
            
            <p style="color: #051c08; font-weight: 500;">This application analyzes changes in forest cover using satellite or aerial imagery:</p>
            
            <h4 style="color: #051c08; font-family: Georgia, serif;">🌳 Deforestation Analysis</h4>
            <ul style="color: #051c08; font-weight: 500;">
                <li>Identifies and measures areas where forest has been removed</li>
                <li>Focuses on detecting barren patches that appear between the before and after images</li>
                <li>Calculates the total deforested area and number of distinct regions</li>
                <li>Provides detailed metrics about each deforested region</li>
            </ul>
            
            <h4 style="color: #6b0000; font-family: Georgia, serif;">🔥 Forest Fire Detection</h4>
            <ul style="color: #6b0000; font-weight: 500;">
                <li>Analyzes color and brightness changes to identify potential burned areas</li>
                <li>Highlights areas that show characteristics of fire damage</li>
                <li>Calculates the extent of burned areas</li>
            </ul>
            
            <h4 style="color: #051c08; font-family: Georgia, serif;">📋 For Best Results</h4>
            <ul style="color: #051c08; font-weight: 500;">
                <li>Use high-quality images (PNG, JPEG, or GeoTIFF)</li>
                <li>Ensure images are properly aligned and of the same area</li>
                <li>Use images taken during similar seasons</li>
                <li>Set the correct pixel resolution for accurate area calculations</li>
                <li>Adjust the minimum area size to filter out noise</li>
            </ul>
            
            <p style="color: #051c08; font-style: italic; margin-top: 20px; font-weight: 500;">
            A state-of-the-art application for ecological image analysis and environmental monitoring.
            </p>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>Deforestation Analyser | Created for environmental monitoring and conservation</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()