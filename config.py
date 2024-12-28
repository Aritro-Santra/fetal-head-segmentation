import os

# Configuration settings
CONFIG = {
    "input_dir": "./data/input",
    "output_dir": "./data/output",
    "roi_coords": None  # Specify ROI coordinates if pre-defined (x, y, width, height)
}

def ensure_directories():
    """Ensure necessary directories exist for output."""
    # Create the main output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Create subdirectories within the output directory
    subdirectories = ["cropped", "filtered", "contours", "segmented", "ground_truth"]
    for subdir in subdirectories:
        os.makedirs(os.path.join(CONFIG["output_dir"], subdir), exist_ok=True)