"""Frozen illumination, motion-blur and Gaussian-noise recipe."""
import math
import hashlib,json
import numpy as np

def degrade(image, condition, field):
    import cv2
    import numpy as np
    i, b, n = condition
    alpha, beta = [(1., 0.), (1.3, 80.), (.8, -50.)][i]
    kernel_size = [1, 7, 21][b]
    variance = [0., 15., 30.][n]
    seed, angle, noise = field
    lit = np.clip(image.astype(np.float32)*alpha+beta, 0, 255).astype(np.uint8)
    if kernel_size == 1:
        blurred = lit
    else:
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        center = (kernel_size-1)/2
        rad = math.radians(angle)
        dx, dy = center*math.cos(rad), center*math.sin(rad)
        cv2.line(kernel, (round(center-dx), round(center-dy)), (round(center+dx), round(center+dy)), 1, 1)
        kernel /= kernel.sum()
        blurred = cv2.filter2D(lit, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    pixels = blurred.copy() if variance == 0 else np.clip(blurred.astype(np.float32)+noise*math.sqrt(variance), 0, 255).astype(np.uint8)
    return pixels, dict(alpha=alpha, beta=beta, kernel=kernel_size, angle=angle, noise_variance=variance,
                        noise_std=math.sqrt(variance), seed=seed)

def field_for(image,stem,seed=20261028):
    key=hashlib.sha256(json.dumps([seed,stem],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    actual=int(key[:16],16);rng=np.random.default_rng(actual)
    return actual,float(rng.uniform(0,180)),rng.standard_normal(image.shape,dtype=np.float32)
