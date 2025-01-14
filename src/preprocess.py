import pydicom
import numpy as np
from pydicom.pixels import apply_voi_lut
from skimage import exposure
from PIL import Image,ImageEnhance
import cv2

def read_xray(path, voi_lut=True, fix_monochrome=True):
    """
    Read and preprocess a DICOM X-ray image.

    Parameters:
    - path: Path to the DICOM file.
    - voi_lut: Apply VOI LUT if available.
    - fix_monochrome: Fix inverted monochrome images.

    Returns:
    - NumPy array: Preprocessed X-ray image.
    """
    dicom = pydicom.dcmread(path)

    # Apply VOI LUT if available
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Fix inverted monochrome images
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Normalize data to start from 0
    data = data - np.min(data)

    return data
    
def resize_pil_image(image: Image.Image, target_shape: tuple) -> Image.Image:
    """
    Resizes a PIL image based on a target shape.

    Args:
        image: Input PIL image.
        target_shape: Desired shape for resizing. It can be a 2D tuple (height, width)
                       or a 3D tuple (height, width, channels), where channels will be ignored.
                       
    Returns:
        Resized PIL image.
    """
    # Convert image to a numpy array
    np_image = np.array(image)

    # Extract the original height and width from the numpy array
    height, width = np_image.shape[:2]

    # If the target shape is 2D (height, width)
    if len(target_shape) == 2:
        new_width, new_height = target_shape
    elif len(target_shape) == 3:
        # If the target shape is 3D (height, width, channels), only change the first two dimensions
        new_width, new_height = target_shape[:2]
    else:
        raise ValueError("Target shape must be either 2D or 3D.")

    # Resize the image using cv2 or PIL's in-built resizing (no channels affected)
    pil_resized_image = Image.fromarray(np_image).resize((new_width, new_height), Image.LANCZOS)

    return pil_resized_image
    
def enhance_exposure(img):
    """
    Enhance image exposure using histogram equalization.

    Parameters:
    - img: Input image as a NumPy array.

    Returns:
    - PIL.Image: Exposure-enhanced image.
    """
    img = exposure.equalize_hist(img)
    img = exposure.equalize_adapthist(img / np.max(img))
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def unsharp_masking(image, kernel_size=5, strength=0.25):
    """
    Apply unsharp masking to enhance image sharpness.

    Parameters:
    - image: Input image as a NumPy array or PIL.Image.
    - kernel_size: Size of the Gaussian blur kernel.
    - strength: Strength of the high-pass filter.

    Returns:
    - PIL.Image: Sharpened image.
    """
    image = np.array(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur and calculate high-pass filter
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    high_pass = cv2.subtract(gray, blurred)

    # Combine high-pass with original image
    sharpened = cv2.addWeighted(gray, 1, high_pass, strength, 0)

    return Image.fromarray(sharpened)

def increase_contrast(image: Image.Image, factor: float) -> Image.Image:
    """
    Increases the contrast of the input PIL image by a given factor.
    
    Args:
        image: Input PIL image.
        factor: Factor by which to increase the contrast. 
                A factor of 1.0 means no change, values greater than 1.0 increase contrast,
                values between 0.0 and 1.0 decrease contrast.
    
    Returns:
        Image with increased contrast.
    """
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(factor)
    return image_enhanced

def increase_brightness(image: Image.Image, factor: float) -> Image.Image:
    """
    Increases the brightness of the input PIL image by a given factor.
    
    Args:
        image: Input PIL image.
        factor: Factor by which to increase the brightness. 
                A factor of 1.0 means no change, values greater than 1.0 increase brightness,
                values between 0.0 and 1.0 decrease brightness.
    
    Returns:
        Image with increased brightness.
    """
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(factor)
    return image_enhanced

def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Parameters:
    - image: Input image as a PIL.Image.
    - clipLimit: Threshold for contrast limiting.
    - tileGridSize: Size of the grid for histogram equalization.

    Returns:
    - Processed image in the same format as the input (PIL.Image).
    """

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    # Apply CLAHE based on image type
    if len(image_np.shape) == 2:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        processed = clahe.apply(image_np)
    else:
        # Color image: Apply CLAHE on the L channel in LAB space
        lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        L_clahe = clahe.apply(L)
        lab_clahe = cv2.merge((L_clahe, A, B))
        processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)

