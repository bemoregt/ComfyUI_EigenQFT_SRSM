"""
QFT Eigenvalue Spectral Residual Saliency Map - ComfyUI Custom Node

Computes a saliency map using the eigenvalues of the local quaternion
cross-spectral matrix (3×3 Hermitian) derived from a Quaternion Fourier
Transform (QFT) of the color image.

Key difference from plain QFT-SR:
  Plain QFT-SR  : uses a simple box filter on the log-amplitude as the prior.
  This node     : uses the dominant eigenvalue λ₁ of the LOCAL 3×3 cross-spectral
                  matrix as the adaptive, color-aware spectral prior.

Algorithm:
  1.  Encode color image: compute per-channel DFTs F_R, F_G, F_B.
  2.  For every frequency (u,v), build the 3×3 Hermitian cross-spectral matrix
      using a k×k local frequency window:
          C[i,j](u,v) = box_filter( F_i · conj(F_j) , size=k )
  3.  Compute eigenvalues λ₁ ≥ λ₂ ≥ λ₃ ≥ 0 of C (batch via numpy.linalg.eigh).
  4.  Spectral residual weight:
          w(u,v) = 1 / sqrt(λ₁(u,v))
      (λ₁ is the dominant energy direction in color-frequency space; dividing
       by sqrt(λ₁) suppresses "predictable" energy and highlights anomalies.)
  5.  Reconstruct each channel:  F_c_SR = w · F_c  , then IFFT.
  6.  Saliency = Σ_c |IFFT(F_c_SR)|² , smoothed with Gaussian.

References:
  - Hou & Zhang (2007). Saliency Detection: A Spectral Residual Approach. CVPR.
  - Schauerte & Stiefelhagen (2012). Quaternion-based Spectral Saliency
    Detection for Eye Fixation Prediction. ECCV.
  - Sangwine & Ell (2000). Hypercomplex Fourier Transforms of Color Images.
"""

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Fallback filters (numpy-only)
# ---------------------------------------------------------------------------

def _uniform_filter_np(arr: np.ndarray, size: int) -> np.ndarray:
    """Box filter via 2-D cumulative sum (integral image trick)."""
    H, W = arr.shape
    pad = size // 2
    a = np.pad(arr, pad, mode="edge")
    cs = np.cumsum(a, axis=0)
    cs = cs[size:] - cs[:-size]
    cs = np.cumsum(cs, axis=1)
    cs = cs[:, size:] - cs[:, :-size]
    return cs / (size * size)


def _gaussian_filter_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter via frequency-domain multiplication."""
    H, W = arr.shape
    fy, fx = np.fft.fftfreq(H), np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    kernel = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (FX ** 2 + FY ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * kernel))


def _uf_real(arr: np.ndarray, size: int) -> np.ndarray:
    if _SCIPY:
        return uniform_filter(arr, size=size)
    return _uniform_filter_np(arr, size)


def _uf_complex(z: np.ndarray, size: int) -> np.ndarray:
    """Box filter applied separately to real and imaginary parts."""
    return _uf_real(z.real, size) + 1j * _uf_real(z.imag, size)


def _gf(arr: np.ndarray, sigma: float) -> np.ndarray:
    if _SCIPY:
        return gaussian_filter(arr, sigma=sigma)
    return _gaussian_filter_np(arr, sigma)


# ---------------------------------------------------------------------------
# Jet colormap (no matplotlib dependency)
# ---------------------------------------------------------------------------

def _jet(gray: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def qft_eigenvalue_spectral_residual(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    window_size: int,
    gaussian_sigma: float,
    eig_index: int = 2,          # 0=λ_min, 1=λ_mid, 2=λ_max (dominant)
) -> np.ndarray:
    """
    Quaternion Eigenvalue Fourier Transform Spectral Residual Saliency.

    Parameters
    ----------
    R, G, B      : (H, W) float64 arrays in [0, 1]
    window_size  : local frequency window for cross-spectral matrix (odd int)
    gaussian_sigma : Gaussian smoothing sigma for the final saliency map
    eig_index    : which eigenvalue to use as prior (0=min, 1=mid, 2=max)

    Returns
    -------
    saliency : (H, W) float32 in [0, 1]
    """
    eps = 1e-24
    k = max(window_size, 3)
    k = k if k % 2 == 1 else k + 1   # ensure odd

    # ------------------------------------------------------------------
    # Step 1: DFT of each channel
    # ------------------------------------------------------------------
    F = [
        np.fft.fft2(R.astype(np.float64)),   # F[0] = F_R
        np.fft.fft2(G.astype(np.float64)),   # F[1] = F_G
        np.fft.fft2(B.astype(np.float64)),   # F[2] = F_B
    ]

    # ------------------------------------------------------------------
    # Step 2: Local 3×3 Hermitian cross-spectral covariance matrix
    #
    #   C[i,j](u,v) = (1/k²) Σ_{|Δu|,|Δv| ≤ k//2} F_i(u+Δu,v+Δv)·F_j*(u+Δu,v+Δv)
    #               = box_filter( F_i · conj(F_j) , size=k )
    #
    # C is positive semi-definite Hermitian  →  real non-negative eigenvalues.
    # ------------------------------------------------------------------
    H, W = R.shape
    C = np.empty((H, W, 3, 3), dtype=np.complex128)

    for i in range(3):
        for j in range(i, 3):
            cij = _uf_complex(F[i] * np.conj(F[j]), k)
            C[:, :, i, j] = cij
            if i != j:
                C[:, :, j, i] = np.conj(cij)

    # ------------------------------------------------------------------
    # Step 3: Batch eigenvalue decomposition
    #   numpy.linalg.eigh  →  ascending eigenvalues (real), valid for Hermitian
    #   Output shape: eigvals (H, W, 3)
    # ------------------------------------------------------------------
    eigvals = np.linalg.eigh(C)[0]          # (..., 3), real, ascending
    lam = np.maximum(eigvals[:, :, eig_index], eps)   # chosen eigenvalue

    # ------------------------------------------------------------------
    # Step 4: Spectral residual weight
    #
    #   Prior amplitude  : A_prior(u,v) = 0.5 · log(λ(u,v))
    #   Total amplitude  : A_total(u,v) = 0.5 · log( Σ_c |F_c|² )
    #   SR               : A_total - A_prior  =  0.5 · log(Σ|F_c|² / λ)
    #
    #   Reconstruction   : F_c_SR = exp(SR) · phase_unit
    #                             = F_c / sqrt(λ)
    #
    #   weight = 1 / sqrt(λ)
    # ------------------------------------------------------------------
    weight = 1.0 / np.sqrt(lam)     # (H, W), real

    # ------------------------------------------------------------------
    # Step 5: Apply weight → inverse DFT per channel
    # ------------------------------------------------------------------
    recon = [np.fft.ifft2(weight * f) for f in F]

    # ------------------------------------------------------------------
    # Step 6: Saliency = Σ_c |reconstructed_c|²
    # ------------------------------------------------------------------
    saliency = sum(np.abs(r) ** 2 for r in recon)

    # ------------------------------------------------------------------
    # Step 7: Gaussian smoothing
    # ------------------------------------------------------------------
    if gaussian_sigma > 0.0:
        saliency = _gf(saliency, gaussian_sigma)

    # ------------------------------------------------------------------
    # Step 8: Normalize to [0, 1]
    # ------------------------------------------------------------------
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency.astype(np.float32)


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class QFTEigenvalueSpectralResidualSaliency:
    """
    QFT Eigenvalue Spectral Residual Saliency Map

    Generates a visual saliency map using the eigenvalues of the local
    3×3 Hermitian cross-spectral matrix derived from the Quaternion
    Fourier Transform (QFT) of a color image.

    Unlike the plain QFT-SR approach (which uses a simple average filter as
    the spectral prior), this node uses the dominant eigenvalue λ₁ of the
    local quaternion cross-spectral matrix as the adaptive prior. λ₁
    represents the maximum "color-correlated" energy at each frequency in
    the local neighborhood, making the prior sensitive to cross-channel
    structure rather than just total power.

    Inputs
    ------
    image         : Input color image (batch supported)
    window_size   : Size of the local frequency window used to build the
                    3×3 cross-spectral matrix. Larger → smoother prior,
                    closer to plain SR. Smaller → more locally adaptive.
    gaussian_sigma: Gaussian blur sigma for the final saliency map.
    eig_mode      : Which eigenvalue to use as the spectral prior.
                    "dominant (λ₁)" — largest eigenvalue: suppresses the
                      strongest color-frequency component (default).
                    "mid (λ₂)"      — middle eigenvalue: highlights mid-range
                      color diversity.
                    "minor (λ₃)"    — smallest eigenvalue: most sensitive to
                      subtle cross-channel anomalies.
    output_mode   : "grayscale" or "heatmap" (jet colormap).

    Output
    ------
    saliency_map : IMAGE, float32 in [0, 1], same size as input.
    """

    _EIG_MAP = {
        "dominant (λ₁)": 2,
        "mid (λ₂)":      1,
        "minor (λ₃)":    0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "window_size": (
                    "INT",
                    {
                        "default": 15,
                        "min": 3,
                        "max": 63,
                        "step": 2,
                        "display": "number",
                    },
                ),
                "gaussian_sigma": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "display": "number",
                    },
                ),
                "eig_mode": (
                    list(cls._EIG_MAP.keys()),
                    {"default": "dominant (λ₁)"},
                ),
                "output_mode": (
                    ["grayscale", "heatmap"],
                    {"default": "grayscale"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saliency_map",)
    FUNCTION = "compute"
    CATEGORY = "image/analysis"

    def compute(
        self,
        image: torch.Tensor,
        window_size: int = 15,
        gaussian_sigma: float = 8.0,
        eig_mode: str = "dominant (λ₁)",
        output_mode: str = "grayscale",
    ):
        """
        Parameters
        ----------
        image : torch.Tensor  shape (B, H, W, C), float32 in [0, 1]

        Returns
        -------
        tuple[torch.Tensor]  shape (B, H, W, 3), float32 in [0, 1]
        """
        eig_idx = self._EIG_MAP.get(eig_mode, 2)
        batch_out = []

        for b in range(image.shape[0]):
            img = image[b].cpu().numpy().astype(np.float64)   # (H, W, C)
            H, W, C = img.shape

            if C >= 3:
                R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            else:
                lum = img[:, :, 0]
                R, G, B = lum, lum, lum

            sal = qft_eigenvalue_spectral_residual(
                R, G, B,
                window_size=window_size,
                gaussian_sigma=gaussian_sigma,
                eig_index=eig_idx,
            )

            if output_mode == "heatmap":
                out = _jet(sal).astype(np.float32)
            else:
                out = np.stack([sal, sal, sal], axis=-1).astype(np.float32)

            batch_out.append(out)

        result = np.stack(batch_out, axis=0)
        return (torch.from_numpy(result),)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "QFTEigenvalueSpectralResidualSaliency": QFTEigenvalueSpectralResidualSaliency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QFTEigenvalueSpectralResidualSaliency": "QFT Eigenvalue SR Saliency",
}
