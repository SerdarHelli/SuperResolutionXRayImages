import gradio as gr
from PIL import Image
from io import BytesIO
from src.pipeline import InferencePipeline
from src.app.config import load_config

# Load configuration and initialize the inference pipeline
config = load_config()
inference_pipeline = InferencePipeline(config)

def process_image_from_bytes(file, apply_clahe_postprocess,apply_pre_contrast_adjustment,return_original_size):
    """
    Process the image bytes using the inference pipeline.

    Args:
        file_bytes: The image file in bytes.
        apply_clahe_postprocess: Boolean indicating if CLAHE postprocessing should be applied.

    Returns:
        The processed image.
    """
    try:
        # Perform super-resolution
        sr_image = inference_pipeline.run(file, apply_pre_contrast_adjustment=apply_pre_contrast_adjustment, apply_clahe_postprocess=apply_clahe_postprocess,return_original_size=return_original_size)
        return sr_image
    except Exception as e:
        return f"An exception occurred: {str(e)}"

# Define the Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # Image Super-Resolution Demo
        Provide image bytes to process and optionally apply CLAHE postprocessing.
        """)

        with gr.Row():
            file_input = gr.File(label="Upload Image (PNG, JPEG, or DICOM)")
            apply_clahe_checkbox = gr.Checkbox(label="Apply CLAHE Postprocessing", value=False)
            apply_preprocess_checkbox = gr.Checkbox(label="Apply PreContrast Adjustment", value=False)
            return_original_size_checkbox = gr.Checkbox(label="Return Original Size", value=True)

        process_button = gr.Button("Process Image")
        output_image = gr.Image(label="Processed Image")



        process_button.click(
            process_image_from_bytes,
            inputs=[file_input, apply_clahe_checkbox,apply_preprocess_checkbox,return_original_size_checkbox],
            outputs=output_image
        )

    return demo

# Launch the Gradio interface
demo = gradio_interface()
demo.queue()


demo.queue(max_size=2)


demo.launch(
    share=False,
    debug=True,
    server_name="0.0.0.0",
    server_port=7860,
    ssl_verify=False,
    auth=[("testoni", "dt1234!")],
)
