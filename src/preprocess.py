import torch
from PIL import Image
import numpy as np
from .preprocess import read_xray, enhance_exposure, unsharp_masking, apply_clahe
from .network.model import RealESRGAN

class Inference:
    def __init__(self, model_weights, device=None, scale=4):
        """
        Initialize the inference pipeline.

        Args:
            model_weights: Path to the model weights file.
            device: Computation device ('cuda' or 'cpu').
            scale: Upscaling factor for the model.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize and load the model
        self.model = RealESRGAN(self.device, scale=scale)
        self.load_weights(model_weights)

    def load_weights(self, model_weights):
        """
        Load the model weights.

        Args:
            model_weights: Path to the model weights file.
        """
        try:
            self.model.load_weights(model_weights)
            print(f"Model weights loaded from {model_weights}")
        except FileNotFoundError:
            print(f"Error: Weight file '{model_weights}' not found.")
        except Exception as e:
            print(f"Error loading weights: {e}")

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
            print(f"Error loading or preprocessing image {image_path}: {e}")
            return None

    def postprocess(self, image_array):
        """
        Postprocess the output from the model.

        Args:
            image_array: Numpy array output from the model.

        Returns:
            PIL Image: Postprocessed image.
        """
        return Image.fromarray(np.uint8(image_array))

    def infer(self, input_image):
        """
        Perform inference on a single image.

        Args:
            input_image: PIL Image to be processed.

        Returns:
            PIL Image: Super-resolved image.
        """
        input_array = np.array(input_image)
        sr_array = self.model.predict(input_array)
        return self.postprocess(sr_array)

    def run(self, input_path, is_dicom=False, apply_clahe_postprocess=False):
        """
        Process a single image and save the output.

        Args:
            input_path: Path to the input image file.
            output_path: Path to save the processed image.
            is_dicom: Boolean indicating if the input is a DICOM file.
            apply_clahe_postprocess: Boolean indicating if CLAHE should be applied post-processing.
        """
        img = self.preprocess(input_path, is_dicom=is_dicom)
        if img is None:
            return

        sr_image = self.infer(img)

        if apply_clahe_postprocess:
            sr_image = apply_clahe(sr_image)

        return sr_image