import streamlit as st
import cv2
import numpy as np
import io
import base64
from PIL import Image
import tempfile
import os
import time
from typing import Tuple, Optional, List

# Set page configuration for better mobile experience
st.set_page_config(
    page_title="Image Processing App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better mobile experience
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
    }
    @media (max-width: 768px) {
        .row-widget.stRadio > div {
            flex-direction: column;
        }
        .row-widget.stRadio > div > label {
            margin-bottom: 10px;
        }
    }
    .uploadedFile {
        margin-bottom: 1rem;
    }
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .css-1aumxhk {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processing_params' not in st.session_state:
    st.session_state.processing_params = {}
if 'current_operation' not in st.session_state:
    st.session_state.current_operation = "Original"

def validate_image(uploaded_file) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate the uploaded image file and return the loaded image if valid.
    
    Args:
        uploaded_file: The uploaded file from Streamlit
        
    Returns:
        Tuple containing:
        - Boolean indicating if validation passed
        - Error message if validation failed, empty string otherwise
        - Loaded image as numpy array if validation passed, None otherwise
    """
    if uploaded_file is None:
        return False, "No file uploaded", None
        
    # Check file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in ['.jpg', '.jpeg', '.png']:
        return False, "Unsupported file format. Please upload JPG, JPEG, or PNG files.", None
    
    # Check file size (limit to 5MB)
    if uploaded_file.size > 5 * 1024 * 1024:
        return False, "File size exceeds 5MB limit. Please upload a smaller image.", None
    
    try:
        # Read image using PIL first (better error handling)
        image = Image.open(uploaded_file)
        
        # Convert to RGB if it has alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert BGR to RGB (OpenCV uses BGR, but we display in RGB)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
        # Check if image dimensions are reasonable
        height, width = img_array.shape[:2]
        if width > 2000 or height > 2000:
            # Resize to reasonable dimensions while maintaining aspect ratio
            scale = min(2000 / width, 2000 / height)
            img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        return True, "", img_array
        
    except Exception as e:
        return False, f"Error processing image: {str(e)}", None

def get_download_link(img, filename="processed_image.png", text="Download Processed Image"):
    """
    Generate a download link for the processed image.
    
    Args:
        img: The processed image as a numpy array
        filename: The filename for the downloaded image
        text: The text to display for the download link
        
    Returns:
        HTML for the download link
    """
    # Convert from BGR to RGB for PIL
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # If grayscale, convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    pil_img = Image.fromarray(img_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    download_link = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px; width: 100%;">{text}</a>'
    return download_link

# Image Processing Functions
def apply_grayscale(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Convert image to grayscale with adjustable intensity.
    
    Args:
        image: Input image as numpy array
        intensity: Intensity level (0.0 to 1.0)
        
    Returns:
        Processed grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply intensity adjustment
    if intensity != 1.0:
        # Create a blended image between original grayscale and adjusted version
        adjusted = cv2.convertScaleAbs(gray, alpha=intensity, beta=0)
        return adjusted
    
    return gray

def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian blur to the image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
        
    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply median blur for noise reduction.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
        
    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_canny_edge(image: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    """
    Apply Canny edge detection.
    
    Args:
        image: Input image
        threshold1: First threshold for the hysteresis procedure
        threshold2: Second threshold for the hysteresis procedure
        
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

def apply_sobel_edge(image: np.ndarray, direction: str = 'both', ksize: int = 3) -> np.ndarray:
    """
    Apply Sobel edge detection.
    
    Args:
        image: Input image
        direction: Direction of the Sobel operator ('x', 'y', or 'both')
        ksize: Size of the Sobel kernel
        
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if direction == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif direction == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:  # both
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize to 0-255
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sobel

def apply_laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    
    Args:
        image: Input image
        ksize: Size of the Laplacian kernel
        
    Returns:
        Edge detected image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    
    # Convert back to uint8
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return laplacian

def apply_dilation(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """
    Apply dilation morphological operation.
    
    Args:
        image: Input image
        kernel_size: Size of the structuring element
        iterations: Number of times dilation is applied
        
    Returns:
        Dilated image
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated

def apply_erosion(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """
    Apply erosion morphological operation.
    
    Args:
        image: Input image
        kernel_size: Size of the structuring element
        iterations: Number of times erosion is applied
        
    Returns:
        Eroded image
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply erosion
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded

def load_anime_style_model():
    """
    Load the pre-trained model for anime style transfer.
    This is a placeholder - in a real app, you would load a specific model.
    
    Returns:
        Loaded model or None if not available
    """
    try:
        # This is where you would load your actual model
        # For demonstration, we'll use a simple OpenCV filter as a placeholder
        return True
    except Exception as e:
        st.error(f"Error loading anime style model: {str(e)}")
        return None

def apply_anime_style_transfer(image: np.ndarray) -> np.ndarray:
    """
    Apply anime style transfer to the image.
    This is a simplified placeholder implementation.
    
    Args:
        image: Input image
        
    Returns:
        Stylized image
    """
    # Placeholder implementation using OpenCV filters
    # In a real app, you would use a properly trained neural network model
    
    # Apply bilateral filter for smoothing while preserving edges
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to grayscale
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create anime-like edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 9, 2)
    
    # Convert edges back to BGR
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine edges with color image
    cartoon = cv2.bitwise_and(smoothed, edges_colored)
    
    # Color quantization to reduce number of colors (anime-like)
    Z = smoothed.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8  # Number of colors
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape(image.shape)
    
    # Blend the edge image with the quantized color image
    result = cv2.addWeighted(cartoon, 0.3, quantized, 0.7, 0)
    
    return result

def display_image_sidebar():
    """Display the sidebar with image upload and processing options."""
    with st.sidebar:
        st.title("Image Processing Toolbox")
        
        # Image upload section
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], 
                                        help="Upload JPG or PNG files (max 5MB)")
        
        if uploaded_file is not None:
            # Validate and load the image
            is_valid, error_msg, img = validate_image(uploaded_file)
            
            if not is_valid:
                st.error(error_msg)
            else:
                # Store the original image in session state
                st.session_state.original_image = img
                
                # Display processing options only if a valid image is uploaded
                display_processing_options()
        else:
            st.info("Please upload an image to start processing")

def display_processing_options():
    """Display the processing options in the sidebar."""
    with st.sidebar:
        st.header("Processing Options")
        
        # Select operation
        operation = st.selectbox(
            "Select Operation",
            ["Original", "Grayscale", "Blur", "Edge Detection", "Morphological Operations", "Anime Style Transfer"]
        )
        
        # Store the current operation in session state
        st.session_state.current_operation = operation
        
        # Display parameters based on the selected operation
        if operation == "Grayscale":
            intensity = st.slider("Intensity", 0.1, 2.0, 1.0, 0.1)
            st.session_state.processing_params = {"intensity": intensity}
            
        elif operation == "Blur":
            blur_type = st.radio("Blur Type", ["Gaussian", "Median", "Bilateral"])
            
            if blur_type == "Gaussian" or blur_type == "Median":
                kernel_size = st.slider("Kernel Size", 1, 25, 5, 2)
                st.session_state.processing_params = {"blur_type": blur_type, "kernel_size": kernel_size}
                
            elif blur_type == "Bilateral":
                d = st.slider("Diameter", 1, 15, 9, 2)
                sigma_color = st.slider("Sigma Color", 1, 150, 75, 1)
                sigma_space = st.slider("Sigma Space", 1, 150, 75, 1)
                st.session_state.processing_params = {
                    "blur_type": blur_type, 
                    "d": d,
                    "sigma_color": sigma_color,
                    "sigma_space": sigma_space
                }
                
        elif operation == "Edge Detection":
            edge_type = st.radio("Edge Detection Type", ["Canny", "Sobel", "Laplacian"])
            
            if edge_type == "Canny":
                threshold1 = st.slider("Threshold 1", 0, 255, 100, 5)
                threshold2 = st.slider("Threshold 2", 0, 255, 200, 5)
                st.session_state.processing_params = {
                    "edge_type": edge_type,
                    "threshold1": threshold1,
                    "threshold2": threshold2
                }
                
            elif edge_type == "Sobel":
                direction = st.radio("Direction", ["x", "y", "both"])
                ksize = st.slider("Kernel Size", 1, 7, 3, 2)
                st.session_state.processing_params = {
                    "edge_type": edge_type,
                    "direction": direction,
                    "ksize": ksize
                }
                
            elif edge_type == "Laplacian":
                ksize = st.slider("Kernel Size", 1, 7, 3, 2)
                st.session_state.processing_params = {
                    "edge_type": edge_type,
                    "ksize": ksize
                }
                
        elif operation == "Morphological Operations":
            morph_type = st.radio("Operation Type", ["Dilation", "Erosion"])
            kernel_size = st.slider("Kernel Size", 1, 15, 3, 2)
            iterations = st.slider("Iterations", 1, 10, 1, 1)
            st.session_state.processing_params = {
                "morph_type": morph_type,
                "kernel_size": kernel_size,
                "iterations": iterations
            }
        
        # Process button
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                process_image()

def process_image():
    """Process the image based on the selected operation and parameters."""
    if st.session_state.original_image is None:
        st.error("No image uploaded")
        return
    
    operation = st.session_state.current_operation
    params = st.session_state.processing_params
    img = st.session_state.original_image.copy()
    
    try:
        # Apply the selected operation
        if operation == "Original":
            processed = img
            
        elif operation == "Grayscale":
            intensity = params.get("intensity", 1.0)
            processed = apply_grayscale(img, intensity)
            
        elif operation == "Blur":
            blur_type = params.get("blur_type", "Gaussian")
            
            if blur_type == "Gaussian":
                kernel_size = params.get("kernel_size", 5)
                processed = apply_gaussian_blur(img, kernel_size)
                
            elif blur_type == "Median":
                kernel_size = params.get("kernel_size", 5)
                processed = apply_median_blur(img, kernel_size)
                
            elif blur_type == "Bilateral":
                d = params.get("d", 9)
                sigma_color = params.get("sigma_color", 75)
                sigma_space = params.get("sigma_space", 75)
                processed = apply_bilateral_filter(img, d, sigma_color, sigma_space)
                
        elif operation == "Edge Detection":
            edge_type = params.get("edge_type", "Canny")
            
            if edge_type == "Canny":
                threshold1 = params.get("threshold1", 100)
                threshold2 = params.get("threshold2", 200)
                processed = apply_canny_edge(img, threshold1, threshold2)
                
            elif edge_type == "Sobel":
                direction = params.get("direction", "both")
                ksize = params.get("ksize", 3)
                processed = apply_sobel_edge(img, direction, ksize)
                
            elif edge_type == "Laplacian":
                ksize = params.get("ksize", 3)
                processed = apply_laplacian_edge(img, ksize)
                
        elif operation == "Morphological Operations":
            morph_type = params.get("morph_type", "Dilation")
            kernel_size = params.get("kernel_size", 3)
            iterations = params.get("iterations", 1)
            
            if morph_type == "Dilation":
                processed = apply_dilation(img, kernel_size, iterations)
            else:  # Erosion
                processed = apply_erosion(img, kernel_size, iterations)
                
        elif operation == "Anime Style Transfer":
            # Show progress for longer operation
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Load model (or placeholder)
            status_text.text("Loading style transfer model...")
            model = load_anime_style_model()
            progress_bar.progress(30)
            
            if model:
                # Apply style transfer
                status_text.text("Applying style transfer...")
                processed = apply_anime_style_transfer(img)
                progress_bar.progress(100)
                status_text.text("Style transfer complete!")
            else:
                st.error("Failed to load anime style model")
                return
            
        # Store the processed image
        st.session_state.processed_image = processed
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def display_main_content():
    """Display the main content area with original and processed images."""
    st.title("Image Processing App")
    
    # Instructions
    with st.expander("How to Use This App", expanded=True):
        st.markdown("""
        ### Instructions:
        1. Upload an image using the sidebar (JPG or PNG, max 5MB)
        2. Select a processing operation from the dropdown
        3. Adjust parameters as needed
        4. Click "Process Image" to apply changes
        5. View results and download processed image if desired
        
        ### Available Operations:
        - **Grayscale**: Convert image to grayscale with adjustable intensity
        - **Blur**: Apply Gaussian, Median, or Bilateral blur
        - **Edge Detection**: Detect edges using Canny, Sobel, or Laplacian methods
        - **Morphological Operations**: Apply Dilation or Erosion
        - **Anime Style Transfer**: Convert images to anime-style artwork
        """)
    
    # Create two columns for displaying original and processed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        if st.session_state.original_image is not None:
            # Convert from BGR to RGB for display
            img_rgb = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_column_width=True)
        else:
            st.info("Upload an image to get started")
    
    with col2:
        st.header("Processed Image")
        if st.session_state.processed_image is not None:
            # Check if the processed image is grayscale or color
            if len(st.session_state.processed_image.shape) == 2 or st.session_state.processed_image.shape[2] == 1:
                # Grayscale image
                st.image(st.session_state.processed_image, use_column_width=True)
            else:
                # Color image - convert from BGR to RGB for display
                processed_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, use_column_width=True)
                
            # Generate and display download link
            st.markdown(
                get_download_link(st.session_state.processed_image), 
                unsafe_allow_html=True
            )
        else:
            if st.session_state.original_image is not None:
                st.info("Select an operation and click 'Process Image'")
            else:
                st.info("Upload an image to see the processed result")

# Main app function
def main():
    # Display the sidebar for image upload and processing options
    display_image_sidebar()
    
    # Display the main content area
    display_main_content()

if __name__ == "__main__":
    main()