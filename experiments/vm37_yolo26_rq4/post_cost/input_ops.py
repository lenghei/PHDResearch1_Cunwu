"""Bounded scratch arrays with unchanged NumPy draw order and arithmetic."""
import math
import hashlib
import numpy as np

VERSION = 'VM37_BLOCK_ARITHMETIC_V1'
BLOCK_ROWS = 128


def clip_inplace(value):
    np.clip(value, 0, 255, out=value)
    return value.astype(np.uint8)


def haze(image, beta, airlight):
    transmission = math.exp(-float(beta))
    value = image.astype(np.float32)
    value *= transmission
    value += (1.0 - transmission) * float(airlight)
    return clip_inplace(value)


def exposure(image, gain, bias):
    value = image.astype(np.float32)
    value *= float(gain)
    value += float(bias)
    return clip_inplace(value)


def gaussian(image, sigma, rng, block_rows=BLOCK_ROWS):
    output = np.empty_like(image)
    for y in range(0, len(image), block_rows):
        part = image[y:y + block_rows]
        value = rng.normal(0., float(sigma), size=part.shape).astype(np.float32)
        value += part.astype(np.float32)
        output[y:y + block_rows] = clip_inplace(value)
    return output


def shot(image, photons, rng, block_rows=BLOCK_ROWS):
    scale = float(photons)
    if scale <= 0:
        raise ValueError('Photon count must be positive.')
    output = np.empty_like(image)
    for y in range(0, len(image), block_rows):
        value = image[y:y + block_rows].astype(np.float32)
        value *= scale / 255.0
        counts = rng.poisson(value)
        value = counts.astype(np.float32)
        value *= 255.0 / scale
        output[y:y + block_rows] = clip_inplace(value)
    return output


from main.training_effects import NativeEffects, FAMILIES, draw, renderer


class ArithmeticEffects(NativeEffects):
    def apply(self, image, image_id, epoch, occurrence):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected native BGR uint8 pixels.')
        output = image.copy()
        trace = []
        ops = renderer()
        for family in FAMILIES:
            if family not in self.recipe['families']:
                continue
            spec = self.recipe['families'][family]
            seed = int.from_bytes(hashlib.sha256(
                f'{self.seed}|{epoch}|{image_id}|{occurrence}|{family}'.encode()).digest()[:8], 'big')
            rng = np.random.default_rng(seed)
            active = bool(rng.random() < spec['probability'])
            event = {'family': family, 'seed': seed, 'probability': spec['probability'],
                     'active': active, 'parameters': {}}
            if active:
                names = {'HazeAug': ('beta', 'airlight'), 'ExposureAug': ('gain', 'bias'),
                         'MotionBlurAug': ('kernel', 'angle_min', 'angle_max'), 'JPEGAug': ('quality',)}
                if family == 'NoiseAug':
                    gaussian_variant = bool(rng.random() < spec['gaussian_probability'])
                    name = 'sigma' if gaussian_variant else 'photons'
                    p = {name: draw(spec, name, rng)}
                    event['variant'] = 'gaussian' if gaussian_variant else 'shot'
                    output = gaussian(output, p[name], rng) if gaussian_variant else shot(output, p[name], rng)
                else:
                    p = {name: draw(spec, name, rng) for name in names[family]}
                    if family == 'HazeAug':
                        output = haze(output, **p)
                    elif family == 'ExposureAug':
                        output = exposure(output, **p)
                    elif family == 'MotionBlurAug':
                        output, extra = ops.apply_motion_blur(output, p['kernel'], p['angle_min'], p['angle_max'], rng)
                        event.update(extra)
                    else:
                        output = ops.apply_jpeg(output, **p)
                event['parameters'] = p
            trace.append(event)
        if output.shape != image.shape or output.dtype != np.uint8:
            raise ValueError('A photometric effect changed geometry.')
        return output, trace
