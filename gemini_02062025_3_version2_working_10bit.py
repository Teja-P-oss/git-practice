"""dithering_10bit_version.py

Author: ChatGPT rewrite – June 4 2025

A 10-bit-aware rewrite of the original “gemini_02062025_3_version2_working.py”.
Main changes
-------------
* **Input bit-depth promoted to 10 bits** (range 0-1023, `np.uint16`).
* **Gradient generator** now spans 0-1023.
* **Quantization utilities** generalized to any `INPUT_BIT_DEPTH`; supports
  2-bit and 4-bit *reductions* (i.e. output bits = `INPUT_BIT_DEPTH – reduction_bits`).
* **Truncation, ordered, random ordered, and FLED dithers** updated for
  arbitrary bit-depth.
* **PSNR calculation** uses full-range 1023.
* **Canvas size fixed to 512×512** (no 256×256 fallback).
* **All returned images** remain `uint16` in the full 0-1023 range; `imshow`
  calls set `vmin`/`vmax` accordingly so the display works even on 8-bit
  monitors (matplotlib rescales internally).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# ------------------------------------------------------------
# Generic bit-depth configuration
# ------------------------------------------------------------
INPUT_BIT_DEPTH: int = 10          # 10-bit input
MAX_VAL: int = (1 << INPUT_BIT_DEPTH) - 1  # 1023 for 10 bits
DTYPE_IMG = np.uint16              # Storage dtype for 10 bits


# ------------------------------------------------------------
# --- LFSR for Randomness (unchanged except dtype tweaks) ---
# ------------------------------------------------------------
class LFSR:
    """Linear-feedback shift register with arbitrary width."""

    def __init__(self, seed: int, taps: tuple[int, ...]):
        self.state = seed or 1  # avoid all-zero state
        self.taps = taps
        self.nbits = max(self.state.bit_length(), *(taps or (8,)))

    def step(self) -> int:
        if not self.taps:
            self.state >>= 1
            return self.state
        feedback = 0
        for tap in self.taps:
            if 0 < tap <= self.nbits:
                feedback ^= (self.state >> (tap - 1)) & 1
        self.state = ((self.state >> 1) | (feedback << (self.nbits - 1))) & (
            (1 << self.nbits) - 1
        )
        return self.state

    def get_random_bits(self, n: int) -> int:
        """Return the *n* LSBs of the next state."""
        return self.step() & ((1 << n) - 1)


# ------------------------------------------------------------
# --- Bayer Matrices (unchanged) ---
# ------------------------------------------------------------
BAYER_2X2 = np.array([[0, 2], [3, 1]], dtype=np.int32)
BAYER_4X4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.int32
)


# ------------------------------------------------------------
# --- Image Generation ---
# ------------------------------------------------------------

def create_gradient_image(height: int, width: int, *, is_rgb: bool = False) -> np.ndarray:
    """Create a horizontal (and optional vertical) gradient spanning 0-MAX_VAL."""
    if is_rgb:
        img = np.zeros((height, width, 3), dtype=DTYPE_IMG)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = (x * MAX_VAL) // (width - 1) if width > 1 else 0
                img[y, x, 1] = (y * MAX_VAL) // (height - 1) if height > 1 else 0
                diag_den = width + height - 2
                img[y, x, 2] = ((x + y) * MAX_VAL) // diag_den if diag_den else 0
    else:
        img = np.zeros((height, width), dtype=DTYPE_IMG)
        for y in range(height):
            line_val = (y * 0)  # placeholder for consistency
            for x in range(width):
                img[y, x] = (x * MAX_VAL) // (width - 1) if width > 1 else 0
    return img


# ------------------------------------------------------------
# --- Quantization Helper (generic bit-depth) ---
# ------------------------------------------------------------

def quantize_value(value: int, reduction_bits: int) -> int:
    """Quantize *value* (0-MAX_VAL) by dropping *reduction_bits* LSBs with rounding."""
    if not (0 <= reduction_bits <= INPUT_BIT_DEPTH):
        raise ValueError("reduction_bits must be within the input bit-depth range")

    if reduction_bits == 0:
        return np.clip(value, 0, MAX_VAL).astype(DTYPE_IMG)

    output_bits = INPUT_BIT_DEPTH - reduction_bits
    levels = 1 << output_bits

    # Round-to-nearest: map to index then back.
    level_idx = (value * (levels - 1) * 2 + MAX_VAL) // (MAX_VAL * 2)
    level_idx = np.clip(level_idx, 0, levels - 1)
    quantized = (level_idx * MAX_VAL * 2 + (levels - 1)) // ((levels - 1) * 2)
    return int(quantized)


# ------------------------------------------------------------
# --- Dithering Algorithms (10-bit capable) ---
# ------------------------------------------------------------

def truncation_dither(img: np.ndarray, reduction_bits: int) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()
    img_int = img.astype(np.int32)
    # Add half-LSB for simple rounding before truncation
    rounding_offset = 1 << (reduction_bits - 1)
    img_int = np.clip(img_int + rounding_offset, 0, MAX_VAL)

    mask = (~((1 << reduction_bits) - 1)) & ((1 << INPUT_BIT_DEPTH) - 1)
    img_int &= mask
    return img_int.astype(DTYPE_IMG)


def ordered_dither(img: np.ndarray, reduction_bits: int, bayer: np.ndarray) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()

    out = np.zeros_like(img, dtype=np.int32)
    h, w = img.shape
    bsz = bayer.shape[0]
    dither_mag = 1 << (reduction_bits - 1)
    max_bayer = bsz * bsz  # values range 0-15 for 4×4

    for y in range(h):
        for x in range(w):
            base = int(img[y, x])
            b_val = int(bayer[y % bsz, x % bsz])
            thresh = ((b_val * (dither_mag * 2 + 1)) // max_bayer) - dither_mag
            out[y, x] = quantize_value(np.clip(base + thresh, 0, MAX_VAL), reduction_bits)
    return out.astype(DTYPE_IMG)


# --- Random Ordered Dithering ---

def _generate_bayer_variants(base_list: list[np.ndarray]) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for M in base_list:
        M = M.astype(np.int32)
        variants.extend([M, np.fliplr(M), np.flipud(M), np.rot90(M)])
    # pad / repeat to 16 variants
    while len(variants) < 16:
        variants.append(variants[len(variants) % len(base_list)])
    return variants[:16]

BASE_RAND_BAYERS = [BAYER_4X4, BAYER_4X4.T]
RAND_VARIANTS = _generate_bayer_variants(BASE_RAND_BAYERS)


def random_ordered_dither(img: np.ndarray, reduction_bits: int, lfsr: LFSR) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()

    out = np.zeros_like(img, dtype=np.int32)
    h, w = img.shape
    dither_mag = 1 << (reduction_bits - 1)
    current_bayer = RAND_VARIANTS[0]

    for y in range(h):
        for x in range(w):
            if x % 4 == 0:
                current_bayer = RAND_VARIANTS[lfsr.get_random_bits(4)]
            bsz = current_bayer.shape[0]
            max_b = bsz * bsz
            b_val = int(current_bayer[y % bsz, x % bsz])
            thresh = ((b_val * (dither_mag * 2 + 1)) // max_b) - dither_mag
            out[y, x] = quantize_value(np.clip(int(img[y, x]) + thresh, 0, MAX_VAL), reduction_bits)
    return out.astype(DTYPE_IMG)


# --- Forward Lookahead Error Diffusion (FLED) ---

def fled_dither(
    img: np.ndarray,
    reduction_bits: int,
    noise_lfsr: LFSR,
    dist_lfsr: LFSR,
    *,
    noise_strength: int = 1,
    lookahead_k: int = 3,
) -> np.ndarray:
    if reduction_bits == 0:
        return img.copy()

    h, w = img.shape
    out = np.zeros_like(img, dtype=DTYPE_IMG)

    # coefficient sets for k=3 and k=4 as before
    coeff_k3 = [
        (8, 0, 0),
        (4, 4, 0),
        (4, 2, 2),
        (2, 4, 2),
    ]
    coeff_k4 = [
        (16, 0, 0, 0),
        (8, 4, 2, 2),
        (8, 8, 0, 0),
        (4, 4, 4, 4),
    ]
    if lookahead_k == 3:
        coeff_sets = coeff_k3
        denom_shift = 3
    elif lookahead_k == 4:
        coeff_sets = coeff_k4
        denom_shift = 4
    else:
        raise ValueError("lookahead_k must be 3 or 4")

    future_err = np.zeros(lookahead_k, dtype=np.int32)

    for y in range(h):
        noise_lfsr.state ^= (y + 1)
        dist_lfsr.state ^= (y + 1)
        future_err.fill(0)
        for x in range(w):
            val = int(img[y, x]) + future_err[0]

            # Triangular dither noise from LFSR (±2 scaled by strength)
            added_noise = 0
            if noise_strength:
                raw = noise_lfsr.get_random_bits(3)
                mapped = [ -2, -1, -1,  0,  0,  1,  1,  2][raw]
                added_noise = mapped * noise_strength

            val_n = np.clip(val + added_noise, 0, MAX_VAL)
            q_val = quantize_value(val_n, reduction_bits)
            out[y, x] = q_val
            q_err = val_n - q_val

            # shift error buffer left
            future_err[:-1] = future_err[1:]
            future_err[-1] = 0

            coeff = coeff_sets[dist_lfsr.get_random_bits(2)]
            for k in range(lookahead_k):
                if x + 1 + k < w:
                    future_err[k] += (q_err * coeff[k]) >> denom_shift
    return out


# ------------------------------------------------------------
# --- Pattern Analysis (unchanged except dtype) ---
# ------------------------------------------------------------

def analyze_patterns(img: np.ndarray) -> dict[str, int]:
    data = img.astype(np.int64)
    h, w = data.shape
    res: dict[str, int] = {}

    # vertical lines
    col_sums = data.sum(axis=0) if w else np.array([])
    if col_sums.size:
        mean = int(col_sums.mean())
        res["vert_line_metric"] = int(np.sqrt(((col_sums - mean) ** 2).mean()))
    else:
        res["vert_line_metric"] = 0

    # horizontal lines
    row_sums = data.sum(axis=1) if h else np.array([])
    if row_sums.size:
        mean_r = int(row_sums.mean())
        res["horiz_line_metric"] = int(np.sqrt(((row_sums - mean_r) ** 2).mean()))
    else:
        res["horiz_line_metric"] = 0

    # texture metrics
    res["texture_col_diff_metric"] = int(np.abs(np.diff(col_sums)).mean()) if col_sums.size > 1 else 0
    res["texture_row_diff_metric"] = int(np.abs(np.diff(row_sums)).mean()) if row_sums.size > 1 else 0
    return res


# ------------------------------------------------------------
# --- Runner / Demo ---
# ------------------------------------------------------------

def run_dithering_comparison(
    *,
    image_height: int = 512,
    image_width: int = 512,
    reduction_bits_list: list[int] | None = None,
) -> None:
    reduction_bits_list = reduction_bits_list or [2, 4]

    lfsr_seed = 0b1010_1010
    lfsr_taps = (8, 6, 5, 4)

    noise_lfsr = LFSR(lfsr_seed, lfsr_taps)
    dist_lfsr = LFSR(lfsr_seed + 1, lfsr_taps)
    rand_lfsr = LFSR(lfsr_seed + 2, lfsr_taps)

    for is_rgb in (False, True):
        grad = create_gradient_image(image_height, image_width, is_rgb=is_rgb)
        img_type = "RGB" if is_rgb else "Grayscale"
        channels = 3 if is_rgb else 1

        print(f"\n=== {img_type} ({image_height}×{image_width}) - 10-bit input ===")
        for rbits in reduction_bits_list:
            out_bits = INPUT_BIT_DEPTH - rbits
            print(f"\n-- Reduction: {rbits} bits  ⇒  {out_bits}-bit output --")
            bayer = BAYER_2X2 if rbits == 2 else BAYER_4X4

            algos = {
                "Original": grad,
                "Trunc": None,
                "Ordered": None,
                "Random": None,
                "FLED K3 S0": None,
                "FLED K3 S1": None,
            }
            psnr_res: dict[str, float] = {}
            patt_res: dict[str, dict[str, int]] = {}

            for name in algos:
                if name == "Original":
                    continue
                out_full = np.zeros_like(grad, dtype=DTYPE_IMG)
                for ch in range(channels):
                    src = grad[..., ch] if is_rgb else grad
                    if name == "Trunc":
                        out = truncation_dither(src, rbits)
                    elif name == "Ordered":
                        out = ordered_dither(src, rbits, bayer)
                    elif name == "Random":
                        out = random_ordered_dither(src, rbits, rand_lfsr)
                    elif name == "FLED K3 S0":
                        out = fled_dither(src, rbits, noise_lfsr, dist_lfsr, noise_strength=0)
                    elif name == "FLED K3 S1":
                        out = fled_dither(src, rbits, noise_lfsr, dist_lfsr, noise_strength=1)
                    else:
                        raise AssertionError

                    if is_rgb:
                        out_full[..., ch] = out
                    else:
                        out_full = out

                algos[name] = out_full

                # metrics
                if is_rgb:
                    psnr_ch = [
                        psnr(grad[..., c], out_full[..., c], data_range=MAX_VAL)
                        for c in range(3)
                    ]
                    psnr_res[name] = sum(psnr_ch) / 3
                    patt_aggr = {k: 0 for k in analyze_patterns(out_full[..., 0])}
                    for c in range(3):
                        p = analyze_patterns(out_full[..., c])
                        for k in patt_aggr:
                            patt_aggr[k] += p[k]
                    for k in patt_aggr:
                        patt_aggr[k] //= 3
                    patt_res[name] = patt_aggr
                else:
                    psnr_res[name] = psnr(grad, out_full, data_range=MAX_VAL)
                    patt_res[name] = analyze_patterns(out_full)

                print(
                    f"  {name:<10}  PSNR={psnr_res[name]:6.2f} dB  "
                    f"V:{patt_res[name]['vert_line_metric']:<4} "
                    f"H:{patt_res[name]['horiz_line_metric']:<4} "
                )

            # --- Visualization (optional) ---
            n_alg = len(algos)
            fig, axes = plt.subplots(2, n_alg, figsize=(n_alg * 3, 6))
            fig.suptitle(
                f"{img_type} – {out_bits}-bit output (10-bit src)", fontsize=14
            )
            zoom = 4
            h_z, w_z = image_height // zoom, image_width // zoom
            y0 = image_height // 2 - h_z // 2
            x0 = image_width // 2 - w_z // 2
            sl = (slice(y0, y0 + h_z), slice(x0, x0 + w_z))

            for idx, (name, im) in enumerate(algos.items()):
                axes[0, idx].imshow(im, cmap="gray" if not is_rgb else None, vmin=0, vmax=MAX_VAL)
                axes[0, idx].set_title(name, fontsize=8)
                axes[0, idx].axis("off")

                zoom_im = im[sl] if channels == 1 else im[sl[0], sl[1], :]
                axes[1, idx].imshow(zoom_im, cmap="gray" if not is_rgb else None, vmin=0, vmax=MAX_VAL, interpolation="nearest")
                axes[1, idx].set_title("zoom×4", fontsize=8)
                axes[1, idx].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


if __name__ == "__main__":
    run_dithering_comparison()

