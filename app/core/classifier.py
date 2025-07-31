import os
import sys
import shutil
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet34_Weights
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import pydicom
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Directory setup - handle both development and packaged environments
if getattr(sys, 'frozen', False):
    # We are running in a bundled app - use consistent location in user's home
    base_dir = Path(os.path.expanduser("~")) / ".pivotal"
    os.makedirs(base_dir, exist_ok=True)
else:
    # We are running in a normal Python environment
    base_dir = Path(__file__).resolve().parent.parent.parent
    print(f"Base directory: {base_dir}")

# For packaged app, use the persistent uploads directory
if getattr(sys, 'frozen', False):
    HOME_DIR = os.path.expanduser("~")
    PERSISTENT_DIR = os.path.join(HOME_DIR, '.pivotal')
    OUTPUT_DIR = os.path.join(PERSISTENT_DIR, 'uploads')
else:
    OUTPUT_DIR = os.path.join(base_dir, 'uploads')

# Create directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created uploads directory in classifier: {OUTPUT_DIR}")
else:
    print(f"Using existing uploads directory: {OUTPUT_DIR}")

TEST_IMAGE_DIR = "Test"
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions_tb")
TEMP_DIR = os.path.join(PREDICTION_DIR, "temp_converted")
base_path = os.path.dirname(os.path.abspath(__file__))
# SEG_MODEL_PATH = os.path.join((base_path), "model_enhanced.pth")
# CLS_MODEL_PATH = os.path.join((base_path), "tb_classifier_v3.pth")
SEG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model_enhanced.pth")
CLS_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tb_classifier_v3.pth")

# Create specific directories for different output types
DICOM_DIR = os.path.join(OUTPUT_DIR, 'dicom')
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'image')
SEGMENTATION_MASKS_DIR = os.path.join(OUTPUT_DIR, 'segmentation_masks')
CAM_OVERLAYS_DIR = os.path.join(OUTPUT_DIR, 'cam_overlays')
HEATMAP_OVERLAYS_DIR = os.path.join(OUTPUT_DIR, 'heatmap_overlays')
ANALYSIS_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'analysis_results')

directories_to_create = [
    (DICOM_DIR, "DICOM directory"),
    (IMAGE_DIR, "image directory"),
    (SEGMENTATION_MASKS_DIR, "segmentation masks directory"),
    (CAM_OVERLAYS_DIR, "CAM overlays directory"),
    (HEATMAP_OVERLAYS_DIR, "heatmap overlays directory"),
    (ANALYSIS_RESULTS_DIR, "analysis results directory"),
    (PREDICTION_DIR, "predictions directory"),
    (TEMP_DIR, "temporary files directory"),
    (os.path.join(OUTPUT_DIR, "models"), "models directory")
]

for directory, description in directories_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created {description}: {directory}")
    else:
        print(f"Using existing {description}: {directory}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_CLS = 224
TEMPERATURE = 1.0
KEEP_CONVERTED = False
BATCH_SIZE = 4
LOG_FILE = os.path.join(PREDICTION_DIR, "predict_tb.log")

WHO_SENSITIVITY = 0.90
WHO_SPECIFICITY = 0.70


val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
])

# Setup logging (ensure log directory exists first)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_name('efficientnet-b0')
        self.base._conv_stem = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc = nn.Linear(self.base._fc.in_features, 2)
        self.base._fc = nn.Identity()
        logger.info("Initialized custom CNN architecture")

    def forward(self, x):
        x = self.base.extract_features(x)
        x = nn.AdaptiveAvgPool2d(1)(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


def get_gradcam_heatmap(model, img_tensor, target_class, target_layer="base._blocks.15"):
    model.eval()
    img_tensor = img_tensor.clone().detach().requires_grad_(True).to(DEVICE)
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    try:
        target_module = dict(model.named_modules())[target_layer]
    except KeyError:
        logger.error(f"Target layer {target_layer} not found in model")
        raise KeyError(f"Target layer {target_layer} not found in model")

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    output = model(img_tensor)
    model.zero_grad()
    output[:, target_class].backward()
    forward_handle.remove()
    backward_handle.remove()

    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    heatmap = torch.sum(weights * activations[0], dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap

def get_scorecam_heatmap(model, input_tensor, target_class, target_layer="base._blocks.15"):
    model.eval()
    input_tensor = input_tensor.clone().detach().to(DEVICE)
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    try:
        target_module = dict(model.named_modules())[target_layer]
    except KeyError:
        logger.error(f"Target layer {target_layer} not found in model")
        raise KeyError(f"Target layer {target_layer} not found in model")

    handle = target_module.register_forward_hook(forward_hook)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    fmap = activations[0]
    n_channels = fmap.shape[1]
    score_weights = []

    for i in range(n_channels):
        fmap_i = fmap[0, i, :, :].cpu().numpy()
        fmap_i_resized = cv2.resize(fmap_i, (input_tensor.shape[3], input_tensor.shape[2]))
        fmap_i_resized = (fmap_i_resized - np.min(fmap_i_resized)) / (np.max(fmap_i_resized) - np.min(fmap_i_resized) + 1e-8)
        fmap_mask = torch.from_numpy(fmap_i_resized).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        masked_input = input_tensor * fmap_mask
        with torch.no_grad():
            score = model(masked_input)[0, target_class].item()
        score_weights.append(score)

    weights = torch.tensor(score_weights).float().to(DEVICE)
    cam = torch.sum(weights.view(1, -1, 1, 1) * fmap, dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

def calculate_per_image_metrics(true_label, pred_class):
    metrics = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'Accuracy': 0.0}
    if true_label == 1 and pred_class == 1:
        metrics['TP'] = 1
        metrics['Accuracy'] = 1.0
    elif true_label == 0 and pred_class == 0:
        metrics['TN'] = 1
        metrics['Accuracy'] = 1.0
    elif true_label == 0 and pred_class == 1:
        metrics['FP'] = 1
    elif true_label == 1 and pred_class == 0:
        metrics['FN'] = 1
    return metrics

def load():
    try:
        seg_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        seg_model.eval()
        logger.info(f"Loaded segmentation model from {SEG_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading segmentation model: {e}")
        raise

    try:
        cls_model = CustomCNN().to(DEVICE)
        cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
        cls_model.eval()
        logger.info(f"Loaded custom CNN classification model from {CLS_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        raise

    return seg_model, cls_model

seg_model, cls_model = load()

def predict(image_path):
    """
    Process a single image for TB classification
    
    Args:
        image_path: Path to the image file (jpg, png, or jpeg)
        
    Returns:
        Dictionary with prediction results and output file paths
    """
    # Load models
    
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Starting prediction on {image_path} using custom CNN")

    results_row = None
    output_files = {
        'segmentation_masks': None,
        'cam_overlays': None,
        'prediction_csvs': None,
        'heatmap_overlays': None
    }
    start_time = time.time()

    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load image {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
        
        base = os.path.basename(image_path).rsplit('.', 1)[0]
        
        # Prepare image for segmentation
        image_seg = cv2.resize(image, (512, 512)) / 255.0
        image_seg = np.stack([image_seg] * 3, axis=-1)  # Stack to 3 channels
        image_tensor = torch.tensor(image_seg, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # Run segmentation
        with torch.no_grad():
            seg_pred = seg_model(image_tensor)
            seg_pred = torch.sigmoid(seg_pred).cpu().numpy().squeeze()
        
        # Process segmentation result
        lung_mask = (seg_pred > 0.5).astype(np.uint8) * 255
        lung_mask = cv2.resize(lung_mask, (image.shape[1], image.shape[0]))
        mask_filename = os.path.join(SEGMENTATION_MASKS_DIR, f"{base}_lung_mask.png")
        cv2.imwrite(mask_filename, lung_mask)
        segmentation_url = "/files/segmentation_masks/"+ f"{base}_lung_mask.png"
        output_files['segmentation_masks'] = segmentation_url

        # Prepare image for classification
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        lung_mask_binary = lung_mask / 255.0
        masked_image = (image_rgb * lung_mask_binary[:, :, np.newaxis]).astype(np.uint8)

        img_pil = Image.fromarray(masked_image).resize((IMG_SIZE_CLS, IMG_SIZE_CLS))
        mask_pil = Image.fromarray(lung_mask).resize((IMG_SIZE_CLS, IMG_SIZE_CLS))

        img_array = np.array(img_pil).astype(np.uint8)
        mask_array = np.expand_dims(np.array(mask_pil).astype(np.float32) / 255.0, axis=2)
        img_masked = np.concatenate([img_array, mask_array * 255], axis=2).astype(np.uint8)

        img_tensor = val_tf(img_masked).unsqueeze(0).to(DEVICE)

        # Run classification
        with torch.no_grad():
            cls_pred = cls_model(img_tensor) / TEMPERATURE
            probs = torch.softmax(cls_pred, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        label = "TB" if pred_class == 1 else "Healthy"

        scorecam_mean = scorecam_max = scorecam_sum = gradcam_mean = severity = affected_percent = None
        if pred_class == 1:
            try:
                scorecam_heatmap = get_scorecam_heatmap(cls_model, img_tensor, target_class=1)
                gradcam_heatmap = get_gradcam_heatmap(cls_model, img_tensor, target_class=1)

                cam_filename = os.path.join(CAM_OVERLAYS_DIR, f"{base}_cam_compare.png")
                scorecam_vis = cv2.applyColorMap((scorecam_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                gradcam_vis = cv2.applyColorMap((gradcam_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                combined = np.concatenate([scorecam_vis, gradcam_vis], axis=1)
                cv2.imwrite(cam_filename, combined)
                cam_url = "/files/cam_overlays/"+ f"{base}_cam_compare.png"
                output_files['cam_overlays'] = cam_url

                scorecam_mean = np.mean(scorecam_heatmap)
                scorecam_max = np.max(scorecam_heatmap)
                scorecam_sum = np.sum(scorecam_heatmap)
                gradcam_mean = np.mean(gradcam_heatmap)
                heatmap = cv2.resize(scorecam_heatmap, (IMG_SIZE_CLS, IMG_SIZE_CLS))

                lung_mask_resized = np.array(mask_pil).astype(np.float32) / 255.0
                lung_mask_binary = (lung_mask_resized > 0.5).astype(np.uint8)
                heatmap_masked = heatmap * lung_mask_binary
                heatmap_masked = heatmap_masked / np.max(heatmap_masked) if np.max(heatmap_masked) > 0 else heatmap_masked

                lung_pixels = heatmap_masked[lung_mask_binary > 0]
                if len(lung_pixels) > 0:
                    lung_pixels_8bit = (lung_pixels * 255).astype(np.uint8)
                    threshold, _ = cv2.threshold(lung_pixels_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    threshold /= 255.0
                    tb_mask = (heatmap_masked >= threshold).astype(np.uint8)
                else:
                    tb_mask = np.zeros_like(heatmap_masked, dtype=np.uint8)

                kernel = np.ones((5, 5), np.uint8)
                tb_mask = cv2.morphologyEx(tb_mask, cv2.MORPH_OPEN, kernel)
                tb_mask = cv2.dilate(tb_mask, kernel, iterations=1)

                affected_area = np.sum(tb_mask * lung_mask_binary)
                total_lung_area = np.sum(lung_mask_binary)
                affected_percent = (affected_area / total_lung_area) * 100 if total_lung_area > 0 else 0

                debug_tb_mask = (tb_mask * 255).astype(np.uint8)
                tb_mask_filename = os.path.join(CAM_OVERLAYS_DIR, f"{base}_tb_binary_mask.png")
                cv2.imwrite(tb_mask_filename, debug_tb_mask)

                if affected_percent > 40:
                    severity = "Severe"
                    severity_color = (255, 0, 0)
                elif affected_percent > 20:
                    severity = "Moderate"
                    severity_color = (0, 0, 255)
                else:
                    severity = "Mild"
                    severity_color = (0, 255, 0)

                heatmap_intensity = (heatmap_masked * 255).astype(np.uint8)
                heatmap_colored = np.zeros((IMG_SIZE_CLS, IMG_SIZE_CLS, 3), dtype=np.uint8)
                for c in range(3):
                    heatmap_colored[:, :, c] = heatmap_intensity * (severity_color[c] / 255.0)
                heatmap_colored = heatmap_colored * lung_mask_binary[:, :, np.newaxis]

                alpha = 0.5
                orig_image_rgb = cv2.resize(image_rgb, (IMG_SIZE_CLS, IMG_SIZE_CLS))
                overlay = cv2.addWeighted(orig_image_rgb, 1.0 - alpha, heatmap_colored, alpha, 0.0)

                contours, _ = cv2.findContours(tb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, severity_color, 1)

                annotation_text = f"{severity} TB ({affected_percent:.1f}%)"
                cv2.putText(overlay, annotation_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, severity_color, 2)

                heatmap_filename = os.path.join(HEATMAP_OVERLAYS_DIR, f"{base}_scorecam_severity.png")
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(heatmap_masked, cmap='jet')
                plt.colorbar(im, label='Heatmap Intensity')
                ax.imshow(orig_image_rgb, alpha=0.5)
                ax.set_title(f"{base} - {severity} TB ({affected_percent:.1f}%)")
                plt.savefig(heatmap_filename)
                plt.close()
                heatmap_url = "/files/heatmap_overlays/"+ f"{base}_scorecam_severity.png"
                output_files['heatmap_overlays'] = heatmap_url
                logger.info(f"Enhanced heatmap saved: {heatmap_filename}")

            except Exception as e:
                logger.error(f"Error computing CAMs for {base}: {e}")
        else:
            severity = "None"
            affected_percent = 0.0

        # Create result dictionary
        results_row = {
            'Image': base,
            'Prediction': label,
            "Prediction_Class": pred_class,
            'Confidence': confidence,
            'ScoreCAM_Mean': scorecam_mean,
            'ScoreCAM_Max': scorecam_max,
            'ScoreCAM_Sum': scorecam_sum,
            'GradCAM_Mean': gradcam_mean,
            'Severity': severity,
            'Affected_Percent': affected_percent
        }

    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

    end_time = time.time()
    
    if not KEEP_CONVERTED and os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory {TEMP_DIR}")

    logger.info(f"Total Time: {end_time - start_time:.2f} seconds")
    
    # Return the results and output files
    return {
        'result': results_row,
        'output_files': output_files
    }

# Load models once at module level