from skimage import io
import numpy as np

def evaluate_iou(ground_truth_path, segmented_path):
    """Compute Intersection over Union (IoU)."""
    gt_img = io.imread(ground_truth_path)
    seg_img = io.imread(segmented_path)

    intersection = np.logical_and(gt_img, seg_img)
    union = np.logical_or(gt_img, seg_img)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def evaluate_dsc(ground_truth_path, segmented_path):
    """Compute Dice Similarity Coefficient (DSC)."""
    gt_img = io.imread(ground_truth_path)
    seg_img = io.imread(segmented_path)

    intersection = np.logical_and(gt_img, seg_img)
    dsc_score = 2 * np.sum(intersection) / (np.sum(gt_img) + np.sum(seg_img))
    return dsc_score

def evaluate_precision(ground_truth_path, segmented_path):
    """Compute Precision."""
    gt_img = io.imread(ground_truth_path)
    seg_img = io.imread(segmented_path)

    true_positive = np.sum(np.logical_and(seg_img == 1, gt_img == 1))
    false_positive = np.sum(np.logical_and(seg_img == 1, gt_img == 0))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    return precision

def evaluate_recall(ground_truth_path, segmented_path):
    """Compute Recall."""
    gt_img = io.imread(ground_truth_path)
    seg_img = io.imread(segmented_path)

    true_positive = np.sum(np.logical_and(seg_img == 1, gt_img == 1))
    false_negative = np.sum(np.logical_and(seg_img == 0, gt_img == 1))
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    return recall

def evaluate_f1(ground_truth_path, segmented_path):
    """Compute F1-Score."""
    precision = evaluate_precision(ground_truth_path, segmented_path)
    recall = evaluate_recall(ground_truth_path, segmented_path)
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score