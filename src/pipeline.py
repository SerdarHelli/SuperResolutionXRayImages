import torch
from PIL import Image
import numpy as np
from io import BytesIO
from huggingface_hub import hf_hub_download
from pathlib import Path

from src.preprocess import read_xray, enhance_exposure, unsharp_masking, apply_clahe, resize_pil_image, increase_brightness
from src.network.model import RealESRGAN
from src.app.exceptions import InputError, ModelLoadError, PreprocessingError, InferenceError,PostprocessingError

class ModelLoadError(Exception):
    pass

class InferencePipeline:
    def __init__(self, config):
        """
        Initialize the inference pipeline using configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        preferred_device = config["model"].get("device", "cuda")
        if preferred_device == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = preferred_device
            
        self.scale = config["model"].get("scale", 4)

        model_source = config["model"].get("source", "local")
        self.model = RealESRGAN(self.device, scale=self.scale)

        print(f"Using device: {self.device}")
        
        try:
            if model_source == "huggingface":
                repo_id = config["model"]["repo_id"]
                filename = config["model"]["filename"]
                local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                self.load_weights(local_path)
            else:
                local_path = config["model"]["weights"]
                self.load_weights(local_path)
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
    def preprocess(self, image_path_or_bytes, apply_pre_contrast_adjustment=True, is_dicom=False):
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
                img = read_xray(image_path_or_bytes)
            else:
                img = Image.open(image_path_or_bytes)
            
            if apply_pre_contrast_adjustment:
                img = enhance_exposure(np.array(img))
                
            if isinstance(img,np.ndarray):
                img = Image.fromarray(((img / np.max(img))*255).astype(np.uint8))   
                
            if img.mode not in ['RGB']:
                img = img.convert('RGB')
                
            img = unsharp_masking(
                img,
                self.config["preprocessing"]["unsharping_mask"].get("kernel_size", 7),
                self.config["preprocessing"]["unsharping_mask"].get("strength", 2)
            )
            img = increase_brightness(
                        img,
                        self.config["preprocessing"]["brightness"].get("factor", 1.2),
                    )

            
            if img.mode not in ['RGB']:
                img = img.convert('RGB')
                
        
            return img, img.size
        except Exception as e:
            raise PreprocessingError(f"Error during preprocessing: {str(e)}")

    def postprocess(self, image_array):
        """
        Postprocess the output from the model.

        Args:
            image_array: PIL.Image output from the model.

        Returns:
            PIL Image: Postprocessed image.
        """
        try:
            return apply_clahe(
                image_array,
                self.config["postprocessing"]["clahe"].get("clipLimit", 2.0),
                tuple(self.config["postprocessing"]["clahe"].get("tileGridSize", [16, 16]))
            )
        except Exception as e:
            raise PostprocessingError(f"Error during postprocessing: {str(e)}")

    def is_dicom(self, file_path_or_bytes):
        """
        Check if the input file is a DICOM file.

        Args:
            file_path_or_bytes (str or bytes or BytesIO): Path to the file, byte content, or BytesIO object.

        Returns:
            bool: True if the file is a DICOM file, False otherwise.
        """
        try:
            if isinstance(file_path_or_bytes, str):
                # Check the file extension
                file_extension = Path(file_path_or_bytes).suffix.lower()
                if file_extension in ['.dcm', '.dicom']:
                    return True

                # Open the file and check the header
                with open(file_path_or_bytes, 'rb') as file:
                    header = file.read(132)
                    return header[-4:] == b'DICM'

            elif isinstance(file_path_or_bytes, BytesIO):
                file_path_or_bytes.seek(0)
                header = file_path_or_bytes.read(132)
                file_path_or_bytes.seek(0)  # Reset the stream position
                return header[-4:] == b'DICM'

            elif isinstance(file_path_or_bytes, bytes):
                header = file_path_or_bytes[:132]
                return header[-4:] == b'DICM'

        except Exception as e:
            print(f"Error during DICOM validation: {e}")
            return False

        return False
        
    def validate_input(self, input_data):
        """
        Validate the input data to ensure it is suitable for processing.

        Args:
            input_data: Path to the input file, bytes content, or BytesIO object.

        Returns:
            bool: True if the input is valid, raises InputError otherwise.
        """
        if isinstance(input_data, str):
            # Check if the file exists
            if not Path(input_data).exists():
                raise InputError(f"Input file '{input_data}' does not exist.")

            # Check if the file type is supported
            file_extension = Path(input_data).suffix.lower()
            if file_extension not in ['.png', '.jpeg', '.jpg', '.dcm', '.dicom']:
                raise InputError(f"Unsupported file type '{file_extension}'. Supported types are PNG, JPEG, and DICOM.")
        
        elif isinstance(input_data, BytesIO):
            # Check if BytesIO data is not empty
            if input_data.getbuffer().nbytes == 0:
                raise InputError("Input BytesIO data is empty.")

        else:
            raise InputError("Unsupported input type. Must be a file path, byte content, or BytesIO object.")

        return True
    
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
        
    def run(self, input_path,  apply_pre_contrast_adjustment = True, apply_clahe_postprocess=False, return_original_size = True):
        """
        Process a single image and save the output.

        Args:
            input_path: Path to the input image file.
            is_dicom: Boolean indicating if the input is a DICOM file.
            apply_clahe_postprocess: Boolean indicating if CLAHE should be applied post-processing.
        """
        # Validate the input
        self.validate_input(input_path)

        is_dicom =self.is_dicom(input_path)

        img, original_size = self.preprocess(input_path, is_dicom=is_dicom, apply_pre_contrast_adjustment = apply_pre_contrast_adjustment)

        if img is None:
            raise InputError(f"Invalid Input")
 

        sr_image = self.infer(img)

        if apply_clahe_postprocess:
            sr_image = self.postprocess(sr_image)

        if return_original_size:
            sr_image = resize_pil_image(sr_image, target_shape = original_size)
            
        return sr_image