import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import time
# import pandas as pd
# import pydicom
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pyvista as pv
import shutil
import logging
from pathlib import Path
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, ResNet34_Weights
import segmentation_models_pytorch as smp
# from .classifier import load as load_classifier

class UNetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e0 = self.relu(self.bn1(self.conv1(x)))
        e1 = self.enc1(self.maxpool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        center = self.center(e4)
        d4 = self.up4(center)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, F.interpolate(e0, size=d1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
        d1 = self.dec1(d1)
        d0 = self.final_up(d1)
        return self.final(d0)

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_name('efficientnet-b0')
        self.base._conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_dim = self.base._fc.in_features
        self.base._fc = nn.Identity()
        self.cls_head = nn.Linear(self.feature_dim, 2)
        self.depth_head = nn.Linear(self.feature_dim, 4)

    def forward(self, x):
        feats = self.base.extract_features(x)
        feats = self.base._avg_pooling(feats)
        feats = feats.flatten(start_dim=1)
        return self.cls_head(feats), self.depth_head(feats)

# Directory setup - handle both development and packaged environments
if getattr(sys, 'frozen', False):
    # We are running in a bundled app - use consistent location in user's home
    base_dir = Path(os.path.expanduser("~")) / ".pivotal"
    os.makedirs(base_dir, exist_ok=True)
else:
    # We are running in a normal Python environment
    base_dir = Path(__file__).resolve().parent.parent.parent

# For packaged app, use the persistent uploads directory
if getattr(sys, 'frozen', False):
    HOME_DIR = os.path.expanduser("~")
    PERSISTENT_DIR = os.path.join(HOME_DIR, '.pivotal')
    OUTPUT_DIR = os.path.join(PERSISTENT_DIR, 'uploads')
else:
    OUTPUT_DIR = os.path.join(base_dir, 'uploads')

# Create directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Created uploads directory in 3d: {OUTPUT_DIR}")
else:
    print(f"Using existing uploads directory: {OUTPUT_DIR}")

# Configuration
TEST_IMAGE_PATH = os.path.join("Test", "0._big_gallery.jpeg")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions_tb")
TEMP_DIR = os.path.join(PREDICTION_DIR, "temp_converted")
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SEG_MODEL_PATH = os.path.join(base_path, "model_enhanced.pth")
# CLS_MODEL_PATH = os.path.join(base_path, "tb_classifier_finetuned_lateral_v2.pth")
SEG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model_enhanced.pth")
CLS_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tb_classifier_finetuned_lateral_v2.pth")

# Create specific directories for different output types
DICOM_DIR = os.path.join(OUTPUT_DIR, 'dicom')
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'image')
SEGMENTATION_MASKS_DIR = os.path.join(OUTPUT_DIR, 'segmentation_masks')
CAM_OVERLAYS_DIR = os.path.join(OUTPUT_DIR, 'cam_overlays')
HEATMAP_OVERLAYS_DIR = os.path.join(OUTPUT_DIR, 'heatmap_overlays')
SEVERITY_OVERLAYS_DIR = os.path.join(OUTPUT_DIR, 'severity_overlays')
RENDERINGS_3D_DIR = os.path.join(OUTPUT_DIR, '3d_renderings')

directories_to_create = [
    (DICOM_DIR, "DICOM directory"),
    (IMAGE_DIR, "image directory"),
    (SEGMENTATION_MASKS_DIR, "segmentation masks directory"),
    (CAM_OVERLAYS_DIR, "CAM overlays directory"),
    (HEATMAP_OVERLAYS_DIR, "heatmap overlays directory"),
    (SEVERITY_OVERLAYS_DIR, "severity overlays directory"),
    (RENDERINGS_3D_DIR, "3D renderings directory"),
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
IMG_SIZE_CLS = 512
TEMPERATURE = 1.0
KEEP_CONVERTED = False
SHOW_3D_WINDOW = False
LOG_FILE = os.path.join(PREDICTION_DIR, "predict_tb_3d.log")

# Lung depth ranges (in mm)
MALE_LUNG_DEPTH_RANGE = (100, 140)
FEMALE_LUNG_DEPTH_RANGE = (80, 120)
DEFAULT_GENDER = "male"
DEPTH_SCALE_FACTOR = IMG_SIZE_CLS / 200

# Ground truth
ground_truth = {"0._big_gallery": "TB"}

# Transforms
val_tf_rgb = transforms.Compose([
    transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Setup logging
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        OUTPUT_DIR,
        PREDICTION_DIR,
        TEMP_DIR,
        os.path.join(OUTPUT_DIR, "models"),
        os.path.join(OUTPUT_DIR, "segmentation_masks"),
        os.path.join(OUTPUT_DIR, "3d_renderings"),
        os.path.join(OUTPUT_DIR, "severity_overlays"),
        os.path.join(OUTPUT_DIR, "cam_overlays"),
        os.path.join(OUTPUT_DIR, "heatmap_overlays")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Ensure directories exist
ensure_directories()

# Setup logging (ensure log directory exists first)
os.makedirs(PREDICTION_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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
    output[0][:, target_class].backward()
    forward_handle.remove()
    backward_handle.remove()

    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    heatmap = torch.sum(weights * activations[0], dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap

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

def estimate_lung_depth(tb_depth, gender=DEFAULT_GENDER):
    depth_range = MALE_LUNG_DEPTH_RANGE if gender == "male" else FEMALE_LUNG_DEPTH_RANGE
    base_depth = depth_range[0] + (depth_range[1] - depth_range[0]) * (tb_depth / 3.0)
    tapered_depth = int(base_depth * DEPTH_SCALE_FACTOR * 0.7)
    return max(tapered_depth, 50)

def create_3d_rendering(image, tb_mask, lung_mask, tb_depth, heatmap, depth_pred, output_path, show_window=SHOW_3D_WINDOW, gender=DEFAULT_GENDER, downsample=1, view=('xy', True), zoom=1.0, roll=90):
    try:
        print(f"Input shapes - image: {image.shape}, tb_mask: {tb_mask.shape}, lung_mask: {lung_mask.shape}, heatmap: {heatmap.shape}")
        image = cv2.resize(image, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_LINEAR)
        tb_mask = cv2.resize(tb_mask, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_NEAREST)
        lung_mask = cv2.resize(lung_mask, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.resize(heatmap, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_LINEAR)
        if downsample > 1:
            new_wh = (image.shape[1] // downsample, image.shape[0] // downsample)
            pad_h = (new_wh[1] * downsample - image.shape[0]) % downsample
            pad_w = (new_wh[0] * downsample - image.shape[1]) % downsample
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                tb_mask = cv2.copyMakeBorder(tb_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                lung_mask = cv2.copyMakeBorder(lung_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                heatmap = cv2.copyMakeBorder(heatmap, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            image = cv2.resize(image, new_wh, interpolation=cv2.INTER_LINEAR)
            tb_mask = cv2.resize(tb_mask, new_wh, interpolation=cv2.INTER_NEAREST)
            lung_mask = cv2.resize(lung_mask, new_wh, interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.resize(heatmap, new_wh, interpolation=cv2.INTER_LINEAR)
        print(f"Resized shapes - image: {image.shape}, tb_mask: {tb_mask.shape}, lung_mask: {lung_mask.shape}, heatmap: {heatmap.shape}")
        image = np.rot90(image, k=-1)
        tb_mask = np.rot90(tb_mask, k=-1)
        lung_mask = np.rot90(lung_mask, k=-1)
        heatmap = np.rot90(heatmap, k=-1)
        lung_mask_binary = (lung_mask / 255.0).astype(np.float32)
        non_lung_mask = 1.0 - lung_mask_binary
        image_lung = image * lung_mask_binary
        tb_mask = tb_mask * lung_mask_binary
        heatmap = heatmap * lung_mask_binary
        adjusted_depth = estimate_lung_depth(tb_depth, gender)
        xray_vol = np.zeros((IMG_SIZE_CLS, IMG_SIZE_CLS, adjusted_depth), dtype=np.float32)
        for z in range(adjusted_depth):
            xray_vol[:, :, z] = image_lung / 255.0
        tb_vol = np.zeros((IMG_SIZE_CLS, IMG_SIZE_CLS, adjusted_depth), dtype=np.float32)
        tb_spread_depth = max(int(adjusted_depth * (tb_depth / 3.0) * 1.0), 1)
        for z in range(min(tb_spread_depth, adjusted_depth)):
            depth_weight = 1.0 - (z / tb_spread_depth) if tb_spread_depth > 1 else 1.0
            tb_vol[:, :, z] = tb_mask * heatmap * depth_weight
        xray_vol = xray_vol * lung_mask_binary[:, :, None]
        tb_vol = tb_vol * lung_mask_binary[:, :, None]
        grid = pv.ImageData(dimensions=xray_vol.shape)
        # print("Xray_vol", xray_vol, xray_vol.shape)
        grid.spacing = (1, 1, 1)
        grid.point_data['XRay'] = xray_vol.ravel(order='F')
        grid.point_data['TB'] = tb_vol.ravel(order='F')
        # Save VTI file with correct extension
        vti_path = os.path.splitext(output_path)[0] + ".vti"

        grid.save(vti_path)
        print(f"Saved VTI file: {vti_path}")
        # mesh = grid.contour(isosurfaces=[0.1], scalars='TB')
        # obj_path = os.path.splitext(output_path)[0] + ".obj"
        # mesh.save(obj_path)
        # xray_mesh = grid.contour(isosurfaces=[0.5], scalars='XRay')
        # tb_mesh = grid.contour(isosurfaces=[0.5], scalars='TB')
        # combined_mesh = xray_mesh + tb_mesh
        # obj_path = os.path.splitext(output_path)[0] + ".obj"
        # combined_mesh.save(obj_path)
        # print(f"3D model saved: {obj_path}")
        # # merged.save(obj_path)
        # # merged.save(os.path.splitext(output_path)[0] + ".vtp")
        # # print(f"Saved combined OBJ: {obj_path}")
        # print(f"Saved OBJ mesh: {obj_path}")
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, scalars='XRay', opacity=[0, 0.8], cmap='gray', clim=[0, 1])
        pl.add_volume(grid, scalars='TB', opacity=[0, 0.5, 0.9], cmap='hot', clim=[0, 1])
        plane, flip = view
        getattr(pl, f'view_{plane}')(negative=flip)
        if zoom != 1.0:
            pl.camera.zoom(zoom)
        if isinstance(roll, (int, float)) and roll != 0:
            pl.camera.Roll(roll)
        # Save screenshot to a PNG file
        png_path = os.path.splitext(output_path)[0] + ".png"
        pl.screenshot(png_path)
        print(f"Saved screenshot to: {png_path}")
        pl.close()
        if show_window:
            pl = pv.Plotter()
            pl.add_volume(grid, scalars='XRay', opacity=[0, 0.8], cmap='gray', clim=[0, 1])
            pl.add_volume(grid, scalars='TB', opacity=[0, 0.5, 0.9], cmap='hot', clim=[0, 1])
            getattr(pl, f'view_{plane}')(negative=flip)
            if zoom != 1.0:
                pl.camera.zoom(zoom)
            pl.show()
        return True
    except Exception as e:
        logger.error(f"Error creating 3D rendering: {e}")
        # Don't re-raise the exception to allow the caller to continue
        return False

def load():
    try:
        seg_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        seg_model.eval()
        logger.info(f"Loaded segmentation model weights from {SEG_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading segmentation model: {e}")
        raise

    try:
        cls_model = CustomCNN().to(DEVICE)
        cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
        cls_model.eval()
        logger.info(f"Loaded custom CNN classification model weights from {CLS_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading custom CNN classification model: {e}")
        raise

    return seg_model, cls_model

seg_model, cls_model = load()

def predict(image_path, pred_class, confidence, gender=DEFAULT_GENDER):
    """
    Generate predictions and 3D rendering for a chest X-ray image.
    
    This function processes an X-ray image to:
    1. Segment the lung regions
    2. Classify the image as TB or healthy
    3. Generate a heatmap showing TB affected areas (if TB is detected)
    4. Create a 3D rendering of the TB-affected lungs (if TB is detected)
    
    Args:
        image_path (str): Path to the image file (PNG, JPG, JPEG only)
        gender (str): Patient's gender ('male' or 'female'), affects the estimated lung depth
    
    Returns:
        list: Results containing prediction details (diagnosis, confidence, TB depth)
    """
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        logger.error(f"Image file not found at {image_path}")
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    # Check if file is an acceptable image format
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in ['.png', '.jpg', '.jpeg']:
        logger.error(f"Unsupported image format: {file_ext}. Only PNG, JPG, and JPEG are supported.")
        raise ValueError(f"Unsupported image format: {file_ext}. Only PNG, JPG, and JPEG are supported.")

    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Starting prediction on {image_path}")

    output_files = {
        'segmentation_mask': None,
        'cam_heatmap': None,
        'severity_overlay': None,
        '3d_rendering': None,
        'prediction_csv': None
    }
    predictions, confidences, results_rows = [], [], []
    start_time = time.time()

    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            raise ValueError(f"Failed to load image: {image_path}")
        
        base = os.path.basename(image_path).rsplit('.', 1)[0]

        # Process the image
        image_seg = cv2.resize(image, (512, 512)) / 255.0
        image_seg = np.stack([image_seg, image_seg, image_seg], axis=-1)
        image_tensor = torch.tensor(image_seg, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            seg_pred = seg_model(image_tensor)
            seg_pred = torch.sigmoid(seg_pred).cpu().numpy().squeeze()
        lung_mask = (seg_pred > 0.5).astype(np.uint8) * 255
        lung_mask = cv2.resize(lung_mask, (image.shape[1], image.shape[0]))
        mask_filename = os.path.join(SEGMENTATION_MASKS_DIR, f"{base}_lung_mask.png")
        cv2.imwrite(mask_filename, lung_mask)
        segmentation_url = "/files/segmentation_masks/"+ f"{base}_lung_mask.png"
        output_files['segmentation_url'] = segmentation_url
        logger.info(f"Segmentation mask saved: {mask_filename}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        lung_mask_binary = lung_mask / 255.0
        masked_image_rgb = (image_rgb * lung_mask_binary[:, :, np.newaxis]).astype(np.uint8)

        img_pil_rgb = Image.fromarray(masked_image_rgb)
        img_tensor_rgb = val_tf_rgb(img_pil_rgb).unsqueeze(0).to(DEVICE)

        img_pil_mask = Image.fromarray(lung_mask).resize((IMG_SIZE_CLS, IMG_SIZE_CLS))
        lung_mask_resized = np.array(img_pil_mask).astype(np.float32) / 255.0
        lung_mask_binary = (lung_mask_resized > 0.5).astype(np.uint8)
        print(pred_class, confidence)
        with torch.no_grad():
            cls_pred, depth_pred = cls_model(img_tensor_rgb)
            probs = torch.softmax(cls_pred / TEMPERATURE, dim=1)[0].cpu().numpy()
        # pred_class = int(np.argmax(probs))
        # confidence = float(probs[pred_class])
        label = "TB" if pred_class == 1 else "Healthy"
        tb_depth = 0

        if pred_class == 1:
            try:
                heatmap = get_gradcam_heatmap(cls_model, img_tensor_rgb, target_class=1)
                heatmap = cv2.resize(heatmap, (IMG_SIZE_CLS, IMG_SIZE_CLS))

                heatmap_masked = heatmap * lung_mask_resized
                heatmap_masked = heatmap_masked / np.max(heatmap_masked) if np.max(heatmap_masked) > 0 else heatmap_masked

                lung_pixels = heatmap_masked[lung_mask_binary > 0]
                if len(lung_pixels) > 0:
                    threshold = np.percentile(lung_pixels, 85)
                    tb_mask = (heatmap_masked > threshold).astype(np.uint8)
                else:
                    tb_mask = np.zeros_like(heatmap_masked, dtype=np.uint8)

                kernel = np.ones((5, 5), np.uint8)
                tb_mask = cv2.morphologyEx(tb_mask, cv2.MORPH_OPEN, kernel)

                affected_area = np.sum(tb_mask * lung_mask_binary)
                total_lung_area = np.sum(lung_mask_binary)
                affected_percent = (affected_area / total_lung_area) * 100 if total_lung_area > 0 else 0

                if affected_percent > 50:
                    tb_depth = 3
                    severity = "Severe"
                    color = (255, 0, 0)
                elif affected_percent > 25:
                    tb_depth = 2
                    severity = "Moderate"
                    color = (0, 0, 255)
                else:
                    tb_depth = 1
                    severity = "Mild"
                    color = (0, 255, 0)

                debug_tb_mask = (tb_mask * 255).astype(np.uint8)
                tb_mask_filename = os.path.join(CAM_OVERLAYS_DIR, f"{base}_tb_binary_mask.png")
                cv2.imwrite(tb_mask_filename, debug_tb_mask)
                cam_url = "/files/cam_overlays/"+ f"{base}_tb_binary_mask.png"
                output_files['cam_url'] = cam_url
                logger.info(f"TB binary mask saved: {tb_mask_filename}")

                color_overlay = np.zeros((IMG_SIZE_CLS, IMG_SIZE_CLS, 3), dtype=np.uint8)
                for i in range(3):
                    color_overlay[:, :, i] = (heatmap_masked * color[i]).astype(np.uint8)

                alpha = 0.4
                orig_image_rgb = cv2.resize(image_rgb, (IMG_SIZE_CLS, IMG_SIZE_CLS))
                overlay = cv2.addWeighted(orig_image_rgb, 1.0 - alpha, color_overlay, alpha, 0.0)

                annotation_text = f"{severity} TB ({affected_percent:.1f}%) - Depth: {tb_depth}"
                cv2.putText(overlay, annotation_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                severity_filename = os.path.join(SEVERITY_OVERLAYS_DIR, f"{base}_heatmap_severity.png")
                cv2.imwrite(severity_filename, overlay)
                severity_url = "/files/severity_overlays/"+ f"{base}_heatmap_severity.png"
                output_files['severity_url'] = severity_url
                logger.info(f"Severity overlay saved: {severity_url}")

                image_resized = cv2.resize(image, (IMG_SIZE_CLS, IMG_SIZE_CLS))
                lung_mask_resized = cv2.resize(lung_mask, (IMG_SIZE_CLS, IMG_SIZE_CLS))
                rendering_filename = os.path.join(RENDERINGS_3D_DIR, f"{base}_3d_rendering.vti")
                rendering_url = "/files/3d_renderings/"+ f"{base}_3d_rendering.vti"
                obj_path = create_3d_rendering(image_resized, tb_mask, lung_mask_resized, tb_depth, heatmap, depth_pred, rendering_filename, gender=gender)
                
                if obj_path:
                    output_files['3d_rendering_url'] = rendering_url
                    output_files['rendering_png_url'] = os.path.splitext(rendering_url)[0] + ".png"
                    logger.info(f"3D rendering saved successfully: {rendering_filename}")
                    print(f"3D rendering saved successfully: {rendering_url}")
                else:
                    logger.warning(f"Failed to create 3D rendering for {base}")
                    print(f"Failed to create 3D rendering for {base}")

            except Exception as e:
                logger.error(f"Heatmap or 3D rendering failed for {base}: {e}")
                severity = "Unknown"
                affected_percent = 0.0
                tb_depth = 0

        base_name = base
        # gt_label = ground_truth.get(base_name)
        
        # if gt_label is not None:
        #     gt_value = 1 if gt_label == "TB" else 0
        predictions.append(pred_class)
        # true_labels.append(gt_value)
        confidences.append(confidence)
        
        # per_image_metrics = calculate_per_image_metrics(gt_value, pred_class)
        
        results_rows.append({
            'Image': base_name,
            'Prediction': label,
            'Confidence': confidence,
            # 'Ground_Truth': gt_label,
            'TB_Depth': tb_depth,
            "output_files": output_files
            # 'TP': per_image_metrics['TP'],
            # 'TN': per_image_metrics['TN'],
            # 'FP': per_image_metrics['FP'],
            # 'FN': per_image_metrics['FN'],
            # 'Accuracy': per_image_metrics['Accuracy']
        })
        
        logger.info(f"Image: {base_name}, Prediction: {label} (Confidence: {confidence:.4f}), TB Depth: {tb_depth}")
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        raise
    
    end_time = time.time()
    
    if not KEEP_CONVERTED and os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory {TEMP_DIR}")
    
    logger.info(f"Total Processed: {len(predictions)}, Total Time: {end_time - start_time:.2f} seconds")
    return results_rows
