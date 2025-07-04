"""
Custom augraphy pipeline for training

This file implements a custom augraphy data augmentation pipeline. We found that using augraphy's
default pipeline can cause significant degradation to formula images, potentially losing semantic
information. Therefore, we carefully selected several common augmentation effects,
adjusting their parameters and combination methods to preserve the original semantic information
of the images as much as possible.
"""

from augraphy import (
    InkColorSwap,
    LinesDegradation,
    OneOf,
    Dithering,
    InkBleed,
    InkShifter,
    NoiseTexturize,
    BrightnessTexturize,
    ColorShift,
    DirtyDrum,
    LightingGradient,
    Brightness,
    Gamma,
    SubtleNoise,
    Jpeg,
    AugraphyPipeline,
)
import random
import numpy as np
import cv2
from collections import Counter

# Scaling ratio for random resizing when training
MAX_RESIZE_RATIO = 1.15
MIN_RESIZE_RATIO = 0.8

MIN_HEIGHT = 30
MIN_WIDTH = 30


def get_aug_pipeline():
    pre_phase = []

    ink_phase = [
        # InkColorSwap(
        #     ink_swap_color="random",
        #     ink_swap_sequence_number_range=(5, 10),
        #     ink_swap_min_width_range=(2, 3),
        #     ink_swap_max_width_range=(100, 120),
        #     ink_swap_min_height_range=(2, 3),
        #     ink_swap_max_height_range=(100, 120),
        #     ink_swap_min_area_range=(10, 20),
        #     ink_swap_max_area_range=(400, 500),
        #     p=0.2,
        # ), # 会出现局部变白的情况
        LinesDegradation(
            line_roi=(0.0, 0.0, 1.0, 1.0),
            line_gradient_range=(32, 255),
            line_gradient_direction=(0, 2),
            line_split_probability=(0.2, 0.4),
            line_replacement_value=(250, 255),
            line_min_length=(30, 40),
            line_long_to_short_ratio=(5, 7),
            line_replacement_probability=(0.4, 0.5),
            line_replacement_thickness=(1, 3),
            p=0.2,
        ),
        #  ============================
        OneOf(
            [
                Dithering(
                    dither="floyd-steinberg",
                    order=(3, 5),
                ),
                InkBleed(
                    intensity_range=(0.1, 0.2),
                    kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
                    severity=(0.4, 0.6),
                ),
            ],
            p=0.2,
        ),
        #  ============================
        #  ============================
        # InkShifter(
        #     text_shift_scale_range=(18, 27),
        #     text_shift_factor_range=(1, 4),
        #     text_fade_range=(0, 2),
        #     blur_kernel_size=(5, 5),
        #     blur_sigma=0,
        #     noise_type="perlin",
        #     p=0.2,
        # ), # MFR任务不应该有扭曲
        #  ============================
    ]

    paper_phase = [
        NoiseTexturize(
            sigma_range=(3, 10),
            turbulence_range=(2, 5),
            texture_width_range=(300, 500),
            texture_height_range=(300, 500),
            p=0.2,
        ),
        BrightnessTexturize(texturize_range=(0.9, 0.99), deviation=0.03, p=0.2),
    ]

    post_phase = [
        ColorShift(
            color_shift_offset_x_range=(3, 5),
            color_shift_offset_y_range=(3, 5),
            color_shift_iterations=(2, 3),
            color_shift_brightness_range=(0.9, 1.1),
            color_shift_gaussian_kernel_range=(3, 3),
            p=0.2,
        ),
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.2,
        ),
        # =====================================
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
                Gamma(
                    gamma_range=(0.9, 1.1),
                ),
            ],
            p=0.2,
        ),
        # =====================================
        # =====================================
        OneOf(
            [
                SubtleNoise(
                    subtle_range=random.randint(5, 10),
                ),
                Jpeg(
                    quality_range=(70, 95),
                ),
            ],
            p=0.2,
        ),
        # =====================================
    ]

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        log=False,
    )

    return pipeline


def trim_white_border(image: np.ndarray, min_len: int = 32) -> np.ndarray:
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            "Image is not in RGB format or channel is not in third dimension"
        )

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    corners = [
        tuple(image[0, 0]),
        tuple(image[0, -1]),
        tuple(image[-1, 0]),
        tuple(image[-1, -1]),
    ]
    bg_color = Counter(corners).most_common(1)[0][0]
    bg_color_np = np.array(bg_color, dtype=np.uint8)

    h, w = image.shape[:2]
    bg = np.full((h, w, 3), bg_color_np, dtype=np.uint8)

    diff = cv2.absdiff(image, bg)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    threshold = 15
    _, diff = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(diff)

    keep_min_h = max(0, min_len - h)
    keep_min_w = max(0, min_len - w)

    trimmed_image = image[y : y + h, x : x + w]

    if keep_min_h > 0 or keep_min_w > 0:
        trimmed_image = cv2.copyMakeBorder(
            trimmed_image,
            top=keep_min_h // 2 + 1,
            bottom=keep_min_h - keep_min_h // 2 + 1,
            left=keep_min_w // 2 + 1,
            right=keep_min_w - keep_min_w // 2 + 1,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return trimmed_image


def random_resize(image: np.ndarray, minr: float, maxr: float) -> np.ndarray:
    ratio = random.uniform(minr, maxr)
    h = max(MIN_HEIGHT, int(image.shape[0] * ratio))
    w = max(MIN_WIDTH, int(image.shape[1] * ratio))
    # Anti-aliasing
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)


def rotate(image: np.ndarray, min_angle: int, max_angle: int) -> np.ndarray:
    # Get the center of the image to define the point of rotation
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # Generate a random angle within the specified range
    angle = random.randint(min_angle, max_angle)

    # Get the rotation matrix for rotating the image around its center
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Determine the size of the rotated image
    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])
    new_width = int((image.shape[0] * sin) + (image.shape[1] * cos))
    new_height = int((image.shape[0] * cos) + (image.shape[1] * sin))

    # Adjust the rotation matrix to take into account translation
    rotation_mat[0, 2] += (new_width / 2) - image_center[0]
    rotation_mat[1, 2] += (new_height / 2) - image_center[1]

    # Rotate the image with the specified border color (white in this case)
    rotated_image = cv2.warpAffine(
        image, rotation_mat, (new_width, new_height), borderValue=(255, 255, 255)
    )

    return rotated_image


def random_pad(image: np.ndarray, pad_len_range: tuple = None) -> np.ndarray:
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            "Image is not in RGB format or channel is not in third dimension"
        )

    if image.dtype != np.uint8:
        raise ValueError(f"Image should stored in uint8")

    if pad_len_range is None:
        max_pad_len = min(image.shape[:2]) // 2
        pad_len_range = (0, max_pad_len)

    pad_len = random.randint(pad_len_range[0], pad_len_range[1])
    pad_h = random.randint(0, pad_len)
    pad_w = random.randint(0, pad_len)
    top = random.randint(0, pad_h)
    bottom = pad_h - top
    left = random.randint(0, pad_w)
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return padded_image
