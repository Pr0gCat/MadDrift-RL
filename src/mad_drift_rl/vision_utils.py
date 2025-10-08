import cv2
import numpy as np
import pytesseract

# Hardcoded floor colors (4 themes in the game)
FLOOR_COLORS = np.array([
    [255, 233,  68],   # Yellow
    [253, 253, 253],  # White
    [172, 116, 197],  # Purple
    [105, 237, 237]   # Cyan
], dtype=np.uint8)

def locate(image: np.ndarray, template: np.ndarray, threshold: float = 0.8, multi_scale: bool = True):
    """
    Find the position of a template image within a larger image using template matching.
    Supports multi-scale matching to handle different window resolutions/DPI settings.

    Args:
        image: Screenshot image as numpy array
        template: Sprite/template to find as numpy array
        threshold: Confidence threshold (0.0 to 1.0), default 0.8
        multi_scale: If True, try multiple scales (0.7x to 1.3x) for robustness

    Returns:
        tuple: (x, y, confidence) of the center position if found, None otherwise
    """
    best_match = None
    best_confidence = threshold

    # Define scale range for multi-scale matching
    scales = np.linspace(0.7, 1.3, 13) if multi_scale else [1.0]

    for scale in scales:
        # Resize template to current scale
        if scale != 1.0:
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)
            # Skip if scaled template is larger than image
            if scaled_h > image.shape[0] or scaled_w > image.shape[1]:
                continue
            scaled_template = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_template = template

        # Perform template matching
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)

        # Find the best match for this scale
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_confidence:
            # Get template dimensions
            h, w = scaled_template.shape[:2]

            # Calculate center position
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2

            best_match = (center_x, center_y, max_val)
            best_confidence = max_val

    return best_match

def ocr(image: np.ndarray) -> str:
    """
    Perform OCR on the given image using pytesseract (optimized for number recognition).

    Args:
        image: Image as numpy array

    Returns:
        Extracted text as string
    """
    # Configure pytesseract for digit-only recognition
    custom_config = r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'

    # Perform OCR
    text = pytesseract.image_to_string(image, config=custom_config)

    return text.strip()

def sample_color(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """
    Sample dominant color from a region of an RGB image.

    Args:
        image: RGB image as numpy array (H, W, 3)
        region: (x, y, width, height) tuple defining the sampling region

    Returns:
        Dominant RGB color as numpy array [R, G, B] with values 0-255
    """
    x, y, w, h = region
    region_pixels = image[y:y+h, x:x+w]
    # Use median for each channel to be robust to outliers
    return np.array([np.median(region_pixels[:, :, i]) for i in range(3)], dtype=np.uint8)

def get_closest_floor_color(sampled_color: np.ndarray) -> np.ndarray:
    """
    Find the closest floor color from hardcoded floor colors.

    Args:
        sampled_color: Sampled RGB color array [R, G, B]

    Returns:
        Closest floor color from FLOOR_COLORS
    """
    # Calculate distances to all floor colors
    distances = np.sqrt(np.sum((FLOOR_COLORS.astype(np.float32) - sampled_color.astype(np.float32)) ** 2, axis=1))
    # Return the closest color
    closest_idx = np.argmin(distances)
    return FLOOR_COLORS[closest_idx]

def normalize_observation(rgb_image: np.ndarray, floor_color: np.ndarray, car_color: np.ndarray) -> np.ndarray:
    """
    Normalize RGB observation to be invariant to environment theme changes.

    This function:
    1. Removes floor pixels in RGB space (sets to black) using closest hardcoded floor color
    2. Creates binary car mask and environment channel separately
    3. Uses morphological operations to clean up the masks
    4. Returns 2-channel output: [car_mask, environment]

    Args:
        rgb_image: RGB image (0-255) as numpy array (H, W, 3)
        floor_color: Sampled floor RGB color as numpy array [R, G, B]
        car_color: Sampled car RGB color as numpy array [R, G, B]

    Returns:
        2-channel image in [0, 1] range as float32 numpy array (H, W, 2)
        - Channel 0: Binary car mask (1.0 for car, 0.0 elsewhere)
        - Channel 1: Environment (CLAHE-normalized grayscale with floor removed, car pixels set to 0.0)
    """
    # Calculate color distance from floor and car colors
    floor_tolerance = 50  # Tolerance for floor color matching in RGB space
    car_tolerance = 40    # Tolerance for car color matching in RGB space

    # Use closest hardcoded floor color
    floor_color = get_closest_floor_color(floor_color)

    # Create floor mask: pixels similar to floor color
    floor_diff = np.sqrt(np.sum((rgb_image.astype(np.float32) - floor_color.astype(np.float32)) ** 2, axis=2))
    floor_mask = floor_diff < floor_tolerance

    # Create car mask: pixels similar to car color
    car_diff = np.sqrt(np.sum((rgb_image.astype(np.float32) - car_color.astype(np.float32)) ** 2, axis=2))
    car_mask = car_diff < car_tolerance

    # Restrict car mask to bottom 10% of the image
    height = rgb_image.shape[0]
    bottom_start = int(height * 0.8)  # Start of bottom 10%
    car_mask[:bottom_start, :] = False  # Zero out everything above bottom 10%

    # Morphological operations to clean up masks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Clean floor mask
    floor_mask = cv2.morphologyEx(floor_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_open)
    floor_mask = cv2.dilate(floor_mask, kernel_dilate, iterations=1)

    # Clean car mask (morphology after restricting to bottom region)
    car_mask = cv2.morphologyEx(car_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, kernel_open)

    # Channel 0: Binary car mask [0, 1]
    car_channel = car_mask.astype(np.float32)

    # Channel 1: Environment (obstacles/walls with floor removed, car pixels set to 0)
    normalized_rgb = rgb_image.copy()
    normalized_rgb[floor_mask.astype(bool)] = [0, 0, 0]  # Floor -> black
    normalized_rgb[car_mask.astype(bool)] = [0, 0, 0]    # Car -> black (will be in separate channel)

    # Convert to grayscale
    gray = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for adaptive contrast normalization on obstacles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Normalize environment channel to [0, 1] range
    env_channel = equalized.astype(np.float32) / 255.0

    # Stack channels: (H, W, 2)
    two_channel = np.stack([car_channel, env_channel], axis=-1)

    return two_channel
