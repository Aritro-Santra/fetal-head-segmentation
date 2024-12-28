import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
from skimage.color import rgba2rgb
from skimage.segmentation import flood_fill

def initialize_contour(x_center, y_center, radius, num_points=400):
    """
    Initialize the contour (snake) for active contour segmentation.

    Args:
        x_center (int): The x-coordinate of the center of the object.
        y_center (int): The y-coordinate of the center of the object.
        radius (int): The radius of the initial contour.
        num_points (int): The number of points in the contour. Default is 400.

    Returns:
        np.ndarray: The initial contour as an array of coordinates.
    """
    s = np.linspace(0, 2 * np.pi, num_points)
    x = x_center + radius * np.cos(s)
    y = y_center + radius * np.sin(s)
    return np.array([x, y]).T

def apply_active_contour(img, init_contour, alpha=0.015, beta=0.85, gamma=0.012,
                         max_iterations=2500, convergence=0.1, boundary_condition='periodic'):
    """
    Apply the active contour model to segment the image.

    Args:
        img (np.ndarray): The input grayscale image.
        init_contour (np.ndarray): The initial contour.
        alpha (float): Elasticity parameter of the contour. Default is 0.015.
        beta (float): Smoothness parameter. Default is 0.85.
        gamma (float): Step size for optimization. Default is 0.012.
        max_iterations (int): Maximum number of iterations. Default is 2500.
        convergence (float): Convergence criterion. Default is 0.1.
        boundary_condition (str): Boundary condition for active contour. Default is 'periodic'.

    Returns:
        np.ndarray: The final contour after active contour segmentation.
    """
    return active_contour(img, init_contour, alpha=alpha, beta=beta, gamma=gamma,
                          max_iterations=max_iterations, convergence=convergence,
                          boundary_condition=boundary_condition, w_line=0, w_edge=1, coordinates='rc')

def save_and_plot_contours(img, init_contour, snake, save_path='contours_output.png'):
    """
    Save and plot the contours on the image.

    Args:
        img (np.ndarray): The input grayscale image.
        init_contour (np.ndarray): The initial contour.
        snake (np.ndarray): The final contour after segmentation.
        save_path (str): Path to save the output image with contours.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init_contour[:, 1], init_contour[:, 0], '-k', lw=400)  # Initial contour
    ax.plot(snake[:, 1], snake[:, 0], '-w', lw=3)  # Final contour
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def flood_fill_segmentation(img, seed_point, tolerance=0.9):
    """
    Apply flood-fill segmentation to the image from a seed point.

    Args:
        img (np.ndarray): The input grayscale image.
        seed_point (tuple): The seed point for the flood-fill operation.
        tolerance (float): The tolerance for the flood-fill operation. Default is 0.9.

    Returns:
        np.ndarray: The filled image after the flood-fill operation.
    """
    return flood_fill(img, seed_point, 127, tolerance=tolerance)

def segment_image(image_path, roi, save_path='contours_output.png', display=False):
    """
    Full segmentation pipeline: Active contour and flood-fill.

    Args:
        image_path (str): Path to the input image.
        roi (tuple): Region of interest (x_center, y_center, radius).
        save_path (str): Path to save the segmented result.
        display (bool): Whether to display intermediate steps.

    Returns:
        np.ndarray: The final segmented image.
    """
    # Load and convert the image to grayscale
    img = io.imread(image_path)
    img_gray = rgb2gray(img) if len(img.shape) == 3 else img

    # Initialize the contour
    x_center, y_center, radius = roi
    init_contour = initialize_contour(x_center, y_center, radius)

    # Apply active contour segmentation
    snake = apply_active_contour(img_gray, init_contour)

    # Save and plot the contours on the image
    save_and_plot_contours(img_gray, init_contour, snake, save_path)

    # Apply flood-fill to segment the object
    filled_image = flood_fill_segmentation(img_gray, seed_point=(y_center, x_center))

    # Optionally display the results
    if display:
        fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
        ax[0].imshow(img_gray, cmap=plt.cm.gray)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(filled_image, cmap=plt.cm.gray)
        ax[1].plot(x_center, y_center, 'ro')  # seed point
        ax[1].set_title('Flood Fill Result')
        ax[1].axis('off')

        plt.show()

    return filled_image
