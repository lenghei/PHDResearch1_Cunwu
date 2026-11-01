"""Deterministic UAV image effects; numeric levels are read from fm_p.uvl."""
from __future__ import annotations
import hashlib
import math
import numpy as np
import cv2

from model import GROUPS, ORDER, recipe, valid_leaves


def effect_seed(master_seed: int, image_stem: str, feature: str) -> int:
    key = f'{master_seed}|{image_stem}|{feature}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')


def clip8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_haze(image, beta, airlight):
    # Constant-depth proxy: this is a synthetic effect, not a reconstructed 3D scene.
    transmission = math.exp(-float(beta))
    return clip8(image.astype(np.float32) * transmission + (1.0 - transmission) * float(airlight))


def apply_rain(image, streaks, stroke_length, opacity, rng):
    height, width = image.shape[:2]
    layer = np.zeros_like(image)
    for _ in range(int(streaks)):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        tail = (min(width - 1, x + max(1, int(stroke_length) // 4)),
                min(height - 1, y + int(stroke_length)))
        cv2.line(layer, (x, y), tail, (215, 215, 215), 1, cv2.LINE_AA)
    return clip8(image.astype(np.float32) + layer.astype(np.float32) * float(opacity))


def apply_exposure(image, gain, bias):
    return clip8(image.astype(np.float32) * float(gain) + float(bias))


def apply_motion_blur(image, kernel_size, angle_min, angle_max, rng):
    size = int(kernel_size)
    if size < 3 or size % 2 != 1:
        raise ValueError('Motion blur kernel must be an odd integer >= 3.')
    angle = float(rng.uniform(float(angle_min), float(angle_max)))
    kernel = np.zeros((size, size), dtype=np.float32)
    radius = (size - 1) / 2
    theta = math.radians(angle)
    dx, dy = radius * math.cos(theta), radius * math.sin(theta)
    cv2.line(kernel, (round(radius - dx), round(radius - dy)),
             (round(radius + dx), round(radius + dy)), 1.0, 1, cv2.LINE_8)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101), {'angle_deg': angle}


def apply_defocus(image, radius):
    radius = int(radius)
    if radius < 1:
        raise ValueError('Defocus radius must be positive.')
    size = radius * 2 + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def apply_rolling_shutter(image, max_shift_fraction, waves, rng):
    h, w = image.shape[:2]
    phase = float(rng.uniform(0, 2 * np.pi))
    rows = np.arange(h, dtype=np.float32)
    shift = w * float(max_shift_fraction) * np.sin(
        (rows / max(1, h - 1)) * 2 * np.pi * int(waves) + phase)
    maps_x = (np.arange(w, dtype=np.float32)[None, :] - shift[:, None]).astype(np.float32)
    maps_y = np.broadcast_to(rows[:, None], (h, w)).copy()
    shifted = cv2.remap(image, maps_x, maps_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    return shifted, {'phase_rad': phase}


def apply_gaussian_noise(image, sigma, rng):
    noise = rng.normal(0.0, float(sigma), size=image.shape).astype(np.float32)
    return clip8(image.astype(np.float32) + noise)


def apply_shot_noise(image, photons, rng):
    scale = float(photons)
    if scale <= 0:
        raise ValueError('Photon count must be positive.')
    counts = rng.poisson(image.astype(np.float32) * (scale / 255.0))
    return clip8(counts.astype(np.float32) * (255.0 / scale))


def apply_downscale(image, factor):
    factor = float(factor)
    if factor <= 1:
        raise ValueError('Downscale factor must be greater than 1.')
    h, w = image.shape[:2]
    size = (max(1, round(w / factor)), max(1, round(h / factor)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_jpeg(image, quality):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError('OpenCV JPEG encoding failed.')
    out = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if out is None:
        raise RuntimeError('OpenCV JPEG decoding failed.')
    return out


# Each FM_P leaf has an explicit renderer. Numeric levels are supplied by UVL.
def leaf_Normal(img, p, rng): return apply_exposure(img, **p), {}
def leaf_Overexposed_mild(img, p, rng): return apply_exposure(img, **p), {}
def leaf_Overexposed_strong(img, p, rng): return apply_exposure(img, **p), {}
def leaf_Underexposed_mild(img, p, rng): return apply_exposure(img, **p), {}
def leaf_Underexposed_strong(img, p, rng): return apply_exposure(img, **p), {}
def leaf_LowLight(img, p, rng): return apply_exposure(img, **p), {}

def leaf_Motion_k7(img, p, rng): return apply_motion_blur(img, p['kernel'], p['angle_min'], p['angle_max'], rng)
def leaf_Motion_k15(img, p, rng): return apply_motion_blur(img, p['kernel'], p['angle_min'], p['angle_max'], rng)
def leaf_Motion_k21(img, p, rng): return apply_motion_blur(img, p['kernel'], p['angle_min'], p['angle_max'], rng)
def leaf_Defocus_r3(img, p, rng): return apply_defocus(img, p['radius']), {}
def leaf_Defocus_r6(img, p, rng): return apply_defocus(img, p['radius']), {}

def leaf_Gauss_s5(img, p, rng): return apply_gaussian_noise(img, p['sigma'], rng), {}
def leaf_Gauss_s15(img, p, rng): return apply_gaussian_noise(img, p['sigma'], rng), {}
def leaf_Gauss_s30(img, p, rng): return apply_gaussian_noise(img, p['sigma'], rng), {}
def leaf_Shot_low(img, p, rng): return apply_shot_noise(img, p['photons'], rng), {}
def leaf_Shot_high(img, p, rng): return apply_shot_noise(img, p['photons'], rng), {}

def leaf_JPEG_q75(img, p, rng): return apply_jpeg(img, p['quality']), {}
def leaf_JPEG_q50(img, p, rng): return apply_jpeg(img, p['quality']), {}
def leaf_JPEG_q25(img, p, rng): return apply_jpeg(img, p['quality']), {}

def leaf_Haze_light(img, p, rng): return apply_haze(img, **p), {}
def leaf_Haze_dense(img, p, rng): return apply_haze(img, **p), {}
def leaf_Rain(img, p, rng): return apply_rain(img, rng=rng, **p), {}

def leaf_Downscale_x1_5(img, p, rng): return apply_downscale(img, p['factor']), {}
def leaf_Downscale_x2(img, p, rng): return apply_downscale(img, p['factor']), {}
def leaf_RollingShutter(img, p, rng): return apply_rolling_shutter(img, rng=rng, **p)

LEAF_RENDERERS = {name: globals()['leaf_' + name] for values in GROUPS.values() for name in values}
STAGE_GROUPS = {
    'atmosphere': ('atmosphere',),
    'illumination': ('illumination',),
    'blur': ('motion', 'defocus'),
    'rolling_shutter': ('rolling',),
    'noise': ('gaussian', 'shot'),
    'downscale': ('altitude',),
    'compression': ('compression',),
}


def generate(image, leaves, attributes, image_stem, master_seed):
    """Return one generated BGR image and per-image, per-leaf provenance."""
    if not valid_leaves(leaves):
        raise ValueError('Invalid selection; refusing to render.')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected original BGR uint8 image.')
    selected = set(leaves)
    parameters = recipe(leaves, attributes)
    result, trace = image.copy(), []
    for stage in ORDER:
        for group in STAGE_GROUPS[stage]:
            for feature in GROUPS[group]:
                if feature not in selected:
                    continue
                seed = effect_seed(master_seed, image_stem, feature)
                result, extra = LEAF_RENDERERS[feature](result, parameters[feature],
                                                       np.random.default_rng(seed))
                trace.append({'feature': feature, 'stage': stage,
                              'params': parameters[feature], 'random_seed': seed, **extra})
    if result.shape != image.shape or result.dtype != np.uint8:
        raise ValueError('Generated image shape or dtype differs from the source.')
    return result, trace
