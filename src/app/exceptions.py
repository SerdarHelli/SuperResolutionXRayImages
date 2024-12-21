class ModelLoadError(Exception):
    """Raised when the model fails to load."""
    def __init__(self, message="Failed to load the model."):
        self.message = message
        super().__init__(self.message)


class PreprocessingError(Exception):
    """Raised when an error occurs during preprocessing."""
    def __init__(self, message="Error during image preprocessing."):
        self.message = message
        super().__init__(self.message)

class PostprocessingError(Exception):
    """Raised when an error occurs during postprocessing."""
    def __init__(self, message="Error during image postprocessing."):
        self.message = message
        super().__init__(self.message)

class InferenceError(Exception):
    """Raised when an error occurs during inference."""
    def __init__(self, message="Error during inference."):
        self.message = message
        super().__init__(self.message)


class InputError(Exception):
    """Raised when an error occurs during loading input."""
    def __init__(self, message="Error loading input."):
        self.message = message
        super().__init__(self.message)