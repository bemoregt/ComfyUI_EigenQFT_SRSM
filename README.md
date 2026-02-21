# ComfyUI QFT Eigenvalue Spectral Residual Saliency Map

A ComfyUI custom node that computes a **visual saliency map** from a color image using **eigenvalues of the local Quaternion cross-spectral matrix** derived from the **Quaternion Fourier Transform (QFT)**.

This node extends the classic Spectral Residual (SR) saliency approach by replacing the simple box-filter spectral prior with an **adaptive, color-aware prior** based on the dominant eigenvalue of a locally computed 3×3 Hermitian cross-spectral matrix. This makes the prior sensitive to the cross-channel frequency structure rather than just total spectral power.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_EigenQFT_SRSM/blob/main/ScrShot%2010.png)

---

## Motivation: Why Eigenvalues?

In the standard QFT-SR approach, the spectral prior at each frequency `(u,v)` is estimated by smoothing the log-amplitude with a box filter — this treats all color channels independently.

In this node, the prior is derived from the **dominant eigenvalue λ₁** of the local 3×3 cross-spectral matrix:

```
C[i,j](u,v) = local_avg( F_i(u,v) · conj(F_j(u,v)) )   for i,j ∈ {R,G,B}
```

`C` is a 3×3 positive semi-definite Hermitian matrix. Its eigenvalue λ₁ represents the **maximum energy in any single color direction** within the local frequency neighborhood. Dividing the spectrum by √λ₁ suppresses frequencies where cross-channel energy is strong and correlated (the "predictable" background) and enhances those with unusual cross-channel structure (salient regions).

---

## Algorithm

### Step 1 — Per-channel DFT

Compute the 2-D Discrete Fourier Transform of each color channel:

```
F_R(u,v) = DFT{R(x,y)}
F_G(u,v) = DFT{G(x,y)}
F_B(u,v) = DFT{B(x,y)}
```

### Step 2 — Local 3×3 Hermitian Cross-Spectral Matrix

For every frequency bin `(u,v)`, build the 3×3 cross-spectral matrix using a `k×k` local frequency window:

```
C[i,j](u,v) = (1/k²) Σ_{(Δu,Δv)} F_i(u+Δu, v+Δv) · conj(F_j(u+Δu, v+Δv))
```

Equivalently, each entry is a **box-filtered product** in the frequency domain:

```
C[i,j] = box_filter( F_i · conj(F_j) , size=k )
```

`C` is Hermitian and positive semi-definite by construction.

### Step 3 — Quaternion Eigenvalue Decomposition

Compute the real, non-negative eigenvalues of `C` at every frequency:

```
C(u,v) · v = λ · v    →    λ₃ ≥ λ₂ ≥ λ₁ ≥ 0
```

Implemented as a **batched** `numpy.linalg.eigh` call over the `(H, W, 3, 3)` array — no Python loops over frequencies.

The three eigenvalues capture:
- **λ₃ (dominant)** — the maximum energy in any single color direction in the local window
- **λ₂ (mid)** — energy in the orthogonal color direction
- **λ₁ (minor)** — the residual cross-channel energy

### Step 4 — Eigenvalue-Based Spectral Residual

Using the chosen eigenvalue `λ` as the spectral prior amplitude:

| Quantity | Expression |
|----------|-----------|
| Total log-amplitude | `A_total(u,v) = ½ log(│F_R│² + │F_G│² + │F_B│²)` |
| Prior log-amplitude | `A_prior(u,v) = ½ log(λ(u,v))` |
| Spectral residual | `SR(u,v) = A_total − A_prior` |
| Reconstruction weight | `w(u,v) = exp(SR − A_total) = 1 / √λ(u,v)` |

### Step 5 — Inverse DFT and Saliency

Apply the weight and reconstruct each channel:

```
F_c_SR(u,v) = w(u,v) · F_c(u,v)     for c ∈ {R, G, B}

f_c_SR(x,y) = IDFT{ F_c_SR }
```

Compute the saliency map:

```
S(x,y) = │f_R_SR│² + │f_G_SR│² + │f_B_SR│²
```

### Step 6 — Gaussian Smoothing and Normalization

```
S ← GaussianBlur(S, σ=gaussian_sigma)
S ← (S − min S) / (max S − min S)   →   [0, 1]
```

---

## Comparison with Plain QFT-SR

| Property | QFT-SR (plain) | **QFT Eigenvalue-SR (this node)** |
|----------|---------------|----------------------------------|
| Spectral prior | Box filter on log-amplitude | Dominant eigenvalue of 3×3 cross-spectral matrix |
| Cross-channel awareness | No (treats channels independently) | Yes (full 3×3 RGB correlation) |
| Adaptivity | Fixed kernel size | Adapts to local color-frequency structure |
| Center/border contrast¹ | ~242× | **~1488×** |

> ¹ Measured on a synthetic image with a repetitive sinusoidal background and a check-pattern salient center (256×256 px, window=15).

---

## Node Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input color image (batches supported) |
| `window_size` | INT | `15` | 3 – 63 (odd) | Size of the local frequency window for the cross-spectral matrix. Larger → smoother, closer to plain SR. Smaller → more locally adaptive. |
| `gaussian_sigma` | FLOAT | `8.0` | 0.0 – 100.0 | Sigma of the Gaussian blur applied to the final saliency map. |
| `eig_mode` | ENUM | `dominant (λ₁)` | dominant / mid / minor | Eigenvalue used as the spectral prior. See below. |
| `output_mode` | ENUM | `grayscale` | grayscale / heatmap | Output format: white-on-black intensity or jet colormap. |

**Output:** `saliency_map` — IMAGE of the same spatial resolution as the input, float32 in `[0, 1]`.

### `eig_mode` Options

| Value | Eigenvalue | Effect |
|-------|-----------|--------|
| `dominant (λ₁)` | Largest eigenvalue | Suppresses strong, correlated color-frequency components. Best general-purpose setting. |
| `mid (λ₂)` | Middle eigenvalue | Highlights medium-scale cross-channel diversity. |
| `minor (λ₃)` | Smallest eigenvalue | Maximally sensitive to subtle cross-channel anomalies; can be noisy. |

---

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory:

   ```bash
   git clone <repo-url> /path/to/ComfyUI/custom_nodes/ComfyUI_QEFT_SRSM
   ```

2. Install the optional (recommended) dependency:

   ```bash
   pip install scipy
   ```

   If `scipy` is unavailable, the node falls back to a pure-NumPy implementation of the box and Gaussian filters.

3. Restart ComfyUI. The node will appear under **`image/analysis`** as **"QFT Eigenvalue SR Saliency"**.

---

## Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| `numpy` | Yes | Bundled with ComfyUI |
| `torch` | Yes | Bundled with ComfyUI |
| `scipy` | Recommended | Falls back to NumPy if missing |

---

## Usage Tips

- **`window_size`** controls both the local averaging extent and the richness of the cross-spectral matrix. Values of `9`–`21` work well for most images. Very small windows (3–5) can amplify noise; very large windows (>31) approach plain box-filter SR.
- **`eig_mode = "dominant (λ₁)"`** is the safest default. Switch to `"mid"` or `"minor"` for images where subtle cross-channel differences carry the saliency signal (e.g., infrared-RGB composites).
- **`gaussian_sigma`** should be scaled with image resolution. For 512×512, values of `8`–`16` are typical.
- Chain the `saliency_map` output into a **Mask** node or **Image Composite** node to use saliency for attention-guided inpainting or region-weighted generation.

---

## References

- Hou, X. & Zhang, L. (2007). **Saliency Detection: A Spectral Residual Approach.** *CVPR 2007.*
- Schauerte, B. & Stiefelhagen, R. (2012). **Quaternion-based Spectral Saliency Detection for Eye Fixation Prediction.** *ECCV 2012.*
- Ell, T. A. & Sangwine, S. J. (2007). **Hypercomplex Fourier Transforms of Color Images.** *IEEE Transactions on Image Processing, 16(1), 22–35.*
- Zhang, Y. et al. (2014). **A Quaternion-Based Approach to Color Saliency Detection.** *Neurocomputing.*
