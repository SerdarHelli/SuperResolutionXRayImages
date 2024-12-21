import torch
from PIL import Image
import numpy as np
import pydicom
from pathlib import Path
import io
from .preprocess import read_xray, enhance_exposure, unsharp_masking, apply_clahe
from .network.model import RealESRGAN
from src.app.exceptions import InputError, ModelLoadError, PreprocessingError, InferenceError,PostprocessingError

class InferencePipeline:
    def __init__(self, config):
        """
        Initialize the inference pipeline using configuration.

        Args:
            config: Configuration dictionary.
        """
        self.device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.scale = config["model"].get("scale", 4)
        self.weights_path = config["model"]["weights"]

        print(f"Using device: {self.device}")
        # Initialize and load the model
        try:
            self.model = RealESRGAN(self.device, scale=self.scale)
            self.load_weights(self.weights_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load the model: {str(e)}")
        
    def load_weights(self, model_weights):
        """
        Load the model weights.

        Args:
            model_weights: Path to the model weights file.
        """
        try:
            self.model.load_weights(model_weights)
        except FileNotFoundError:
            raise ModelLoadError(f"Model weights not found at '{model_weights}'.")
        except Exception as e:
            raise ModelLoadError(f"Error loading weights: {str(e)}")

    def preprocess(self, image_path, is_dicom=False):
        """
        Preprocess the input image.

        Args:
            image_path: Path to the input image file.
            is_dicom: Boolean indicating if the input is a DICOM file.

        Returns: 
            PIL Image: Preprocessed image.
        """
        try:
            if is_dicom:
                img = read_xray(image_path)
                img = enhance_exposure(img)
                img = unsharp_masking(img, 7, 0.5)
                return img
            else:
                img = Image.open(image_path).convert('RGB')
                img = enhance_exposure(np.array(img))
                img = unsharp_masking(img, 7, 0.5)
                return img
        except Exception as e:
            raise PreprocessingError(f"Error during postprocessing: {str(e)}")

    def postprocess(self, image_array):
        """
        Postprocess the output from the model.

        Args:
            image_array: Numpy array output from the model.

        Returns:
            PIL Image: Postprocessed image.
        """
        try:
            return apply_clahe(image_array,clipLimit=2.0, tileGridSize=(16,16))
        except Exception as e:
            raise PostprocessingError(f"Error during inference: {str(e)}")

    def is_dicom(self,file_path_or_bytes):
        """
        Check if the input file is a DICOM file.

        Args:
            file_path_or_bytes (str or bytes): Path to the file or byte content of the file.

        Returns:
            bool: True if the file is a DICOM file, False otherwise.
        """
        try:
            # If input is a file path
            if isinstance(file_path_or_bytes, str):
                # Check file extension
                file_extension = Path(file_path_or_bytes).suffix.lower()
                if file_extension in ['.dcm', '.dicom']:
                    return True
                
                # Check DICOM header
                with open(file_path_or_bytes, 'rb') as file:
                    header = file.read(132)  # DICOM files have a 128-byte preamble followed by "DICM"
                    is_dicom_header = header[-4:] == b'DICM'
                    if is_dicom_header:
                        pydicom.dcmread(file_path_or_bytes)  # Ensure it's a valid DICOM
                    return is_dicom_header

            # If input is byte content
            if isinstance(file_path_or_bytes, bytes):
                header = file_path_or_bytes[:132]
                is_dicom_header = header[-4:] == b'DICM'
                if is_dicom_header:
                    pydicom.dcmread(io.BytesIO(file_path_or_bytes))  # Ensure it's a valid DICOM
                return is_dicom_header

        except Exception:
            return False

    def infer(self, input_image):
        """
        Perform inference on a single image.

        Args:
            input_image: PIL Image to be processed.

        Returns:
            PIL Image: Super-resolved image.
        """
        try:
            # Perform inference
            input_array = np.array(input_image)
            sr_array = self.model.predict(input_array)
            return sr_array
        
        except Exception as e:
            raise InferenceError(f"Error during inference: {str(e)}")
        
    def run(self, input_path,  apply_clahe_postprocess=False):
        """
        Process a single image and save the output.

        Args:
            input_path: Path to the input image file.
            is_dicom: Boolean indicating if the input is a DICOM file.
            apply_clahe_postprocess: Boolean indicating if CLAHE should be applied post-processing.
        """

        is_dicom =self.is_dicom(input_path)

        img = self.preprocess(input_path, is_dicom=is_dicom)

        if img is None:
            raise InputError(f"Invalid Input")
 

        sr_image = self.infer(img)

        if apply_clahe_postprocess:
            sr_image = apply_clahe(sr_image)

        return sr_image