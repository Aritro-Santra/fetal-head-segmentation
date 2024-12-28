import cv2
import numpy as np
import random
from skimage import io

def gamma_correction(img, gamma=0.09):
    """
    Apply gamma correction to the input image.

    Args:
        img (np.ndarray): The input image.
        gamma (float): The gamma value for correction. Default is 0.09.

    Returns:
        np.ndarray: The gamma-corrected image.
    """
    return np.array(255 * (img / 255) ** gamma, dtype='uint8')

def add_salt_and_pepper_noise(img, p=0.3):
    """
    Add salt and pepper noise to the image.

    Args:
        img (np.ndarray): The input image.
        p (float): The probability of adding noise. Default is 0.3.

    Returns:
        np.ndarray: The noisy image.
    """
    rows, columns, channels = img.shape
    noisy_img = np.zeros(img.shape, np.uint8)
    for i in range(rows):
        for j in range(columns):
            r = random.random()
            if r < p / 2:
                noisy_img[i][j] = [0, 0, 0]  # Salt (Black)
            elif r < p:
                noisy_img[i][j] = [255, 255, 255]  # Pepper (White)
            else:
                noisy_img[i][j] = img[i][j]
    return noisy_img

def denoise_image(img):
    """
    Apply a series of denoising filters to the image:
    - Median blur (3 iterations)
    - Gaussian blur
    - Bilateral filter

    Args:
        img (np.ndarray): The noisy input image.

    Returns:
        np.ndarray: The denoised image.
    """
    # Median Blur (3 iterations)
    median_blur = cv2.medianBlur(img, 9)
    median_blur_1 = cv2.medianBlur(median_blur, 9)
    median_blur_2 = cv2.medianBlur(median_blur_1, 9)
    
    # Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(median_blur_2, (9, 9), 0)
    
    # Bilateral Filter
    bilateral_blur = cv2.bilateralFilter(gaussian_blur, 9, 20, 100, borderType=cv2.BORDER_CONSTANT)
    
    return bilateral_blur

def convert_to_grayscale(img):
    """
    Convert the image to grayscale.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_image(image_path, display_images=False):
    """
    Preprocess the input image by applying a series of transformations:
    - Gamma correction
    - Salt and pepper noise addition
    - Denoising with median blur, Gaussian blur, and bilateral filter
    - Grayscale conversion

    Args:
        image_path (str): Path to the input image.
        display_images (bool): Whether to display images at each step. Default is False.

    Returns:
        np.ndarray: The processed grayscale image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Apply gamma correction
    gamma_corrected = gamma_correction(img)
    if display_images:
        io.imshow(gamma_corrected)
        io.show()

    # Add salt and pepper noise
    noisy_img = add_salt_and_pepper_noise(gamma_corrected)
    if display_images:
        io.imshow(noisy_img)
        io.show()

    # Apply denoising filters
    denoised_img = denoise_image(noisy_img)
    if display_images:
        io.imshow(denoised_img)
        io.show()

    # Convert to grayscale
    img_gray = convert_to_grayscale(denoised_img)
    if display_images:
        io.imshow(img_gray, cmap='gray')
        io.show()

    return img_gray
