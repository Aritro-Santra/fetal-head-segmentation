import cv2
import os

def select_roi(image_path, predefined_coords=None):
    """
    Manually select a Region of Interest (ROI) or use predefined coordinates.

    Args:
        image_path (str): Path to the input image.
        predefined_coords (tuple, optional): Predefined coordinates (x, y, width, height) for the ROI. 
                                             If not provided, user will be prompted to select ROI manually.

    Returns:
        tuple: The cropped image and the ROI coordinates (x, y, width, height).
    """
    # Read the input image
    img = cv2.imread(image_path)
    
    # Use predefined coordinates if provided, otherwise let the user select ROI manually
    if predefined_coords:
        roi = predefined_coords
    else:
        roi = cv2.selectROI("Select ROI", img)
        cv2.destroyAllWindows()

    # Ensure the ROI is valid
    x, y, w, h = map(int, roi)

    # Crop the image to the selected ROI
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img, roi

def save_image(output_subdir, filename, image, output_dir="./data/output"):
    """
    Save an image to a specific output subdirectory.

    Args:
        output_subdir (str): Subdirectory under the output directory to save the image.
        filename (str): The name of the file to save the image as.
        image (np.ndarray): The image to be saved.
        output_dir (str, optional): The main output directory. Defaults to './data/output'.

    Returns:
        str: The full path where the image is saved.
    """
    # Create the full output path
    output_path = os.path.join(output_dir, output_subdir)
    
    # Check if the subdirectory exists, create it if it doesn't
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create the full path to save the image
    full_output_path = os.path.join(output_path, filename)
    
    # Save the image
    cv2.imwrite(full_output_path, image)
    
    return full_output_path