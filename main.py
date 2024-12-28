import argparse
from src.config import CONFIG, ensure_directories
from src.utils import select_roi, save_image
from src.preprocessing.py import preprocess_image
from src.segmentation import segment_image
from src.evaluation import evaluate_iou
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Image Segmentation Pipeline.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--ground_truth", required=True, help="Path to the ground truth image.")
    args = parser.parse_args()

    ensure_directories()

    cropped_img, roi = select_roi(args.image, CONFIG["roi_coords"])
    cropped_path = save_image("cropped", "cropped_image.png", cropped_img)

    preprocessed_img = preprocess_image(cropped_path)
    filtered_path = save_image("filtered", "filtered_image.png", preprocessed_img)

    segmented_gray, snake = segment_image(filtered_path, roi)
    plt.figure()
    plt.imshow(segmented_gray, cmap="gray")
    plt.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
    segmented_path = os.path.join(CONFIG["output_dir"], "segmented", "segmented_image.png")
    plt.savefig(segmented_path)
    plt.close()

    iou_score = evaluate_iou(args.ground_truth, segmented_path)
    print(f"IoU Score: {iou_score:.4f}")

    print("Pipeline completed successfully.")
