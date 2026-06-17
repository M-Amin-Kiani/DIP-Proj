# ============================================================
# Morphological operations on a real photo (Python / Colab)
# ------------------------------------------------------------
# Colab installation cell (run once before this script):
# !pip install -q numpy matplotlib scipy scikit-image pillow opencv-python-headless
# ============================================================

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    black_tophat,
    closing,
    convex_hull_image,
    dilation,
    disk,
    erosion,
    opening,
    skeletonize,
    thin,
    white_tophat,
)
from skimage.segmentation import clear_border
from skimage.util import img_as_float, img_as_ubyte


# ------------------------------------------------------------
# 0) User image path
# ------------------------------------------------------------
# In Colab, upload your photo first, or set this to the uploaded filename.
IMAGE_PATH = Path("photo_2024-02-06_03-54-07.jpg")

if not IMAGE_PATH.exists():
    try:
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        if len(uploaded) == 0:
            raise FileNotFoundError("No file uploaded.")
        IMAGE_PATH = Path(next(iter(uploaded.keys())))
    except Exception as e:
        raise FileNotFoundError(
            "Could not find the input image. Put the photo next to the notebook "
            "or upload it in Colab and try again."
        ) from e


# ------------------------------------------------------------
# 1) Output folders
# ------------------------------------------------------------
OUT_DIR = Path("morphology_real_photo_outputs")
OUT_DIR.mkdir(exist_ok=True)


def save_fig(fig, name: str):
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_binary_image(arr, name: str):
    img = (arr.astype(np.uint8) * 255)
    Image.fromarray(img).save(OUT_DIR / name)


def save_gray_image(arr, name: str):
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(OUT_DIR / name)


# ------------------------------------------------------------
# 2) Load image and prepare grayscale
# ------------------------------------------------------------
def load_rgb_and_gray(path: Path):
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
    gray_f = img_as_float(gray)
    gray_eq = exposure.equalize_adapthist(gray_f, clip_limit=0.02)
    return rgb, gray_f, gray_eq


# ------------------------------------------------------------
# 3) Foreground segmentation with GrabCut
# ------------------------------------------------------------
def grabcut_foreground_mask(rgb: np.ndarray):
    """Return a cleaned binary foreground mask for the person + instrument."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Rectangle chosen for this specific photo.
    # It encloses the main subject while leaving some room for GrabCut.
    rect = (60, 35, w - 90, h - 60)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 0/2 -> background, 1/3 -> foreground
    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)

    # Clean small artifacts and fill holes
    bin_mask = ndi.binary_closing(bin_mask, structure=np.ones((5, 5), dtype=bool))
    bin_mask = ndi.binary_opening(bin_mask, structure=np.ones((3, 3), dtype=bool))
    bin_mask = ndi.binary_fill_holes(bin_mask)
    bin_mask = clear_border(bin_mask)

    # Keep the largest connected component (the main subject)
    lab = label(bin_mask)
    regions = regionprops(lab)
    if len(regions) > 0:
        largest = max(regions, key=lambda r: r.area).label
        bin_mask = lab == largest

    return bin_mask


# ------------------------------------------------------------
# 4) Helper utilities
# ------------------------------------------------------------
def largest_component(mask: np.ndarray):
    lab = label(mask)
    regions = regionprops(lab)
    if not regions:
        return mask.copy()
    largest = max(regions, key=lambda r: r.area).label
    return lab == largest


def bbox_of_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def overlay_boundary_on_gray(gray: np.ndarray, boundary: np.ndarray):
    rgb = np.dstack([gray, gray, gray]).copy()
    rgb[boundary, 0] = 1.0
    rgb[boundary, 1] = 0.0
    rgb[boundary, 2] = 0.0
    return rgb


def endpoint_mask_from_skeleton(skel: np.ndarray):
    """Detect endpoints on a skeleton using a 3x3 neighborhood count."""
    conv = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
    # Endpoint: foreground pixel with exactly one foreground neighbor.
    # conv includes the center, so endpoint count = 2.
    endpoints = skel & (conv == 2)
    return endpoints


# ------------------------------------------------------------
# 5) Operations on the real photo
# ------------------------------------------------------------
def op_smoothing_and_gradient(binary_mask: np.ndarray):
    se = disk(5)
    smoothed = closing(opening(binary_mask, footprint=se), footprint=se)
    grad = dilation(smoothed, footprint=disk(3)).astype(np.int16) - erosion(smoothed, footprint=disk(3)).astype(np.int16)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Original binary mask")
    axes[1].imshow(smoothed, cmap="gray")
    axes[1].set_title("Morphological smoothing")
    axes[2].imshow(grad, cmap="gray")
    axes[2].set_title("Morphological gradient")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "smooth_gradient_seed_3.png")
    return smoothed, grad


def op_binary_dilations(binary_mask: np.ndarray):
    # Kept the previous filenames for compatibility, but now they mean
    # three different structuring-element sizes on the real photo.
    configs = [
        ("dilation_run_7.png", 3),
        ("dilation_run_42.png", 7),
        ("dilation_run_2025.png", 13),
    ]
    for fname, r in configs:
        out = dilation(binary_mask, footprint=disk(r))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(out, cmap="gray")
        ax.set_title(f"Binary dilation, disk r={r}")
        ax.axis("off")
        plt.tight_layout()
        save_fig(fig, fname)


def op_erosion(binary_mask: np.ndarray):
    radii = [3, 7, 13, 19]
    outs = [erosion(binary_mask, footprint=disk(r)) for r in radii]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Original binary mask")
    axes[0].axis("off")
    for ax, out, r in zip(axes[1:], outs, radii):
        ax.imshow(out, cmap="gray")
        ax.set_title(f"Erosion r={r}")
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "erosion_results.png")
    return outs


# ------------------------------------------------------------
# 6) Top-hat / Bottom-hat on grayscale
# ------------------------------------------------------------
def op_tophat_bottomhat(gray_eq: np.ndarray):
    se = disk(17)
    top = white_tophat(gray_eq, footprint=se)
    bottom = black_tophat(gray_eq, footprint=se)

    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    axes1[0].imshow(gray_eq, cmap="gray")
    axes1[0].set_title("Enhanced grayscale image")
    axes1[0].axis("off")
    axes1[1].imshow(top, cmap="gray")
    axes1[1].set_title("White top-hat")
    axes1[1].axis("off")
    plt.tight_layout()
    save_fig(fig1, "tophat_results.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    axes2[0].imshow(gray_eq, cmap="gray")
    axes2[0].set_title("Enhanced grayscale image")
    axes2[0].axis("off")
    axes2[1].imshow(bottom, cmap="gray")
    axes2[1].set_title("Black top-hat / bottom-hat")
    axes2[1].axis("off")
    plt.tight_layout()
    save_fig(fig2, "bottomhat_results.png")

    return top, bottom


# ------------------------------------------------------------
# 7) Flat gray-scale morphology
# ------------------------------------------------------------
def op_gray_flat(gray_eq: np.ndarray):
    se = disk(9)
    ero = erosion(gray_eq, footprint=se)
    dil = dilation(gray_eq, footprint=se)
    opn = opening(gray_eq, footprint=se)
    cls = closing(gray_eq, footprint=se)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].imshow(gray_eq, cmap="gray")
    axes[0].set_title("Original grayscale")
    axes[1].imshow(ero, cmap="gray")
    axes[1].set_title("Erosion (flat disk)")
    axes[2].imshow(dil, cmap="gray")
    axes[2].set_title("Dilation (flat disk)")
    axes[3].imshow(opn, cmap="gray")
    axes[3].set_title("Opening")
    axes[4].imshow(cls, cmap="gray")
    axes[4].set_title("Closing")
    axes[5].axis("off")
    for ax in axes[:-1]:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "gray_flat_morphology.png")
    return ero, dil, opn, cls


# ------------------------------------------------------------
# 8) Nonflat gray-scale morphology
# ------------------------------------------------------------
def op_gray_nonflat(gray_eq: np.ndarray):
    # A bowl-shaped nonflat structuring element
    r = 7
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    d2 = x**2 + y**2
    structure = -0.02 * d2
    footprint = d2 <= r**2
    structure = np.where(footprint, structure, 0.0)

    ero_nf = ndi.grey_erosion(gray_eq, footprint=footprint, structure=structure)
    dil_nf = ndi.grey_dilation(gray_eq, footprint=footprint, structure=structure)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(gray_eq, cmap="gray")
    axes[0].set_title("Original grayscale")
    axes[1].imshow(ero_nf, cmap="gray")
    axes[1].set_title("Nonflat erosion")
    axes[2].imshow(dil_nf, cmap="gray")
    axes[2].set_title("Nonflat dilation")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "gray_nonflat_morphology.png")
    return ero_nf, dil_nf


# ------------------------------------------------------------
# 9) Boundary extraction
# ------------------------------------------------------------
def op_boundary_extraction(binary_mask: np.ndarray, gray_eq: np.ndarray):
    boundary = binary_mask & (~erosion(binary_mask, footprint=disk(2)))
    overlay = overlay_boundary_on_gray(gray_eq, boundary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Binary mask")
    axes[1].imshow(boundary, cmap="gray")
    axes[1].set_title("Extracted boundary")
    axes[2].imshow(overlay)
    axes[2].set_title("Boundary overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "boundary_extraction.png")
    return boundary


# ------------------------------------------------------------
# 10) Connected component experiments
# ------------------------------------------------------------
def op_connected_components(binary_mask: np.ndarray):
    lab = label(binary_mask)
    props = regionprops(lab)
    count = len(props)

    # All components
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    axes1[0].imshow(binary_mask, cmap="gray")
    axes1[0].set_title("Binary mask")
    axes1[1].imshow(lab, cmap="nipy_spectral")
    axes1[1].set_title(f"Connected components (n={count})")
    axes1[2].imshow(binary_mask, cmap="gray")
    axes1[2].set_title("Input for CCE")
    for ax in axes1:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig1, "connected_component_result.png")

    # Largest component only
    largest = largest_component(binary_mask)
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    axes2[0].imshow(lab, cmap="nipy_spectral")
    axes2[0].set_title("Component labels")
    axes2[0].axis("off")
    axes2[1].imshow(largest, cmap="gray")
    axes2[1].set_title("Largest connected component")
    axes2[1].axis("off")
    plt.tight_layout()
    save_fig(fig2, "cce_result.png")

    # Bounding boxes of the top few components
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
    ax3.imshow(binary_mask, cmap="gray")
    ax3.set_title("CCE result 2: bounding boxes of the main components")
    for r in sorted(props, key=lambda r: r.area, reverse=True)[:8]:
        minr, minc, maxr, maxc = r.bbox
        ax3.add_patch(
            plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=1.5)
        )
        ax3.text(minc, minr - 2, str(r.label), color="yellow", fontsize=9)
    ax3.axis("off")
    plt.tight_layout()
    save_fig(fig3, "cce_result2.png")

    return lab, largest


# ------------------------------------------------------------
# 11) Region filling
# ------------------------------------------------------------
def op_region_filling(binary_mask: np.ndarray):
    filled = ndi.binary_fill_holes(binary_mask)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Before region filling")
    axes[1].imshow(filled, cmap="gray")
    axes[1].set_title("After region filling")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "region_filling.png")
    return filled


# ------------------------------------------------------------
# 12) Convex hull
# ------------------------------------------------------------
def op_convex_hull(binary_mask: np.ndarray):
    hull = convex_hull_image(binary_mask)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Original binary mask")
    axes[1].imshow(hull, cmap="gray")
    axes[1].set_title("Convex hull")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "convex_hull_example1.png")
    return hull


# ------------------------------------------------------------
# 13) Set operations on binary masks
# ------------------------------------------------------------
def op_set_operations(binary_mask: np.ndarray):
    # Create a second mask by shifting + slightly dilating the foreground.
    shifted = np.roll(binary_mask, shift=22, axis=1)
    shifted = ndi.binary_dilation(shifted, structure=np.ones((5, 5), dtype=bool))

    union = binary_mask | shifted
    inter = binary_mask & shifted
    diff = binary_mask & (~shifted)
    xor = binary_mask ^ shifted

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()
    ax[0].imshow(binary_mask, cmap="gray")
    ax[0].set_title("A")
    ax[1].imshow(shifted, cmap="gray")
    ax[1].set_title("B")
    ax[2].imshow(union, cmap="gray")
    ax[2].set_title("A ∪ B")
    ax[3].imshow(inter, cmap="gray")
    ax[3].set_title("A ∩ B")
    ax[4].imshow(diff, cmap="gray")
    ax[4].set_title("A - B")
    ax[5].imshow(xor, cmap="gray")
    ax[5].set_title("A XOR B")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    save_fig(fig, "set_operations_demo.png")
    return shifted, union, inter, diff, xor


# ------------------------------------------------------------
# 14) Hit-or-miss
# ------------------------------------------------------------
def op_hit_or_miss(binary_mask: np.ndarray):
    # Use the skeleton to detect endpoints in a practical way.
    skel = skeletonize(binary_mask)
    endpoint = endpoint_mask_from_skeleton(skel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(skel, cmap="gray")
    axes[0].set_title("Skeleton")
    axes[1].imshow(endpoint, cmap="gray")
    axes[1].set_title("Hit-or-miss style endpoint map")
    overlay = np.dstack([skel.astype(float)] * 3)
    overlay[endpoint, 0] = 1.0
    overlay[endpoint, 1] = 0.0
    overlay[endpoint, 2] = 0.0
    axes[2].imshow(overlay)
    axes[2].set_title("Endpoint overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "hitmiss_results.png")
    return skel, endpoint


# ------------------------------------------------------------
# 15) Skeletonization / thinning / pruning / thickening
# ------------------------------------------------------------
def prune_skeleton(skel: np.ndarray, iterations: int = 10):
    out = skel.copy()
    for _ in range(iterations):
        ep = endpoint_mask_from_skeleton(out)
        out = out & (~ep)
        if not out.any():
            break
    return out


def op_skeletonization(binary_mask: np.ndarray):
    skel = skeletonize(binary_mask)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Original binary mask")
    axes[1].imshow(skel, cmap="gray")
    axes[1].set_title("Skeletonization result")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "skeletonization_result.png")
    return skel


def op_thinning(binary_mask: np.ndarray):
    # Show a demo with intermediate thinning and final thinning.
    thin5 = thin(binary_mask, max_num_iter=5)
    thin_final = thin(binary_mask)

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    axes1[0].imshow(binary_mask, cmap="gray")
    axes1[0].set_title("Input mask")
    axes1[1].imshow(thin5, cmap="gray")
    axes1[1].set_title("Thinning demo (5 iters)")
    axes1[2].imshow(thin_final, cmap="gray")
    axes1[2].set_title("Thinning demo (final)")
    for ax in axes1:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig1, "thinning_demo.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    axes2[0].imshow(binary_mask, cmap="gray")
    axes2[0].set_title("Original mask")
    axes2[1].imshow(thin_final, cmap="gray")
    axes2[1].set_title("Thinning result")
    for ax in axes2:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig2, "thinning_result.png")

    return thin_final


def op_thickening(binary_mask: np.ndarray):
    # Practical thickening-style growth of a thin structure.
    edge = binary_mask & (~erosion(binary_mask, footprint=disk(1)))
    thick = edge.copy()
    for _ in range(3):
        thick = dilation(thick, footprint=disk(1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(edge, cmap="gray")
    axes[0].set_title("Seed edge / thin structure")
    axes[1].imshow(thick, cmap="gray")
    axes[1].set_title("Thickening demo")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "thickening_demo.png")
    return thick


def op_pruning(binary_mask: np.ndarray):
    skel = skeletonize(binary_mask)
    pruned = prune_skeleton(skel, iterations=12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(binary_mask, cmap="gray")
    axes[0].set_title("Input mask")
    axes[1].imshow(skel, cmap="gray")
    axes[1].set_title("Skeleton")
    axes[2].imshow(pruned, cmap="gray")
    axes[2].set_title("Pruning demo")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "pruning_demo.png")
    return pruned


# ------------------------------------------------------------
# 16) Granulometry
# ------------------------------------------------------------
def op_granulometry(binary_mask: np.ndarray):
    radii = list(range(1, 22, 2))
    areas = []
    opened_imgs = []
    total = float(binary_mask.sum())

    for r in radii:
        opn = opening(binary_mask, footprint=disk(r))
        opened_imgs.append(opn)
        areas.append(opn.sum() / total if total > 0 else 0.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    ax = axes.ravel()
    ax[0].imshow(binary_mask, cmap="gray")
    ax[0].set_title("Original mask")
    ax[0].axis("off")
    for i, (r, img_) in enumerate(zip(radii[:7], opened_imgs[:7]), start=1):
        ax[i].imshow(img_, cmap="gray")
        ax[i].set_title(f"Opening r={r}")
        ax[i].axis("off")
    ax[7].plot(radii, areas)
    ax[7].set_title("Granulometry curve")
    ax[7].set_xlabel("disk radius")
    ax[7].set_ylabel("normalized area")
    plt.tight_layout()
    save_fig(fig, "granulometry_results.png")
    return radii, areas


# ------------------------------------------------------------
# 17) Textural segmentation (derived from the same photo)
# ------------------------------------------------------------
def op_textural_segmentation(gray_eq: np.ndarray):
    # Local standard deviation acts as a texture map derived from the photo.
    win = 21
    mean = ndi.uniform_filter(gray_eq, size=win)
    mean2 = ndi.uniform_filter(gray_eq ** 2, size=win)
    local_std = np.sqrt(np.maximum(mean2 - mean ** 2, 0.0))
    tex = exposure.rescale_intensity(local_std, out_range=(0, 1))

    thr = threshold_otsu(tex)
    tex_bin = tex > thr
    tex_closed = closing(tex_bin, footprint=disk(5))
    tex_opened = opening(tex_closed, footprint=disk(7))
    grad = dilation(tex_opened, footprint=disk(3)).astype(np.int16) - erosion(tex_opened, footprint=disk(3)).astype(np.int16)
    boundary = grad > 0

    overlay = np.dstack([gray_eq, gray_eq, gray_eq]).copy()
    overlay[boundary, 0] = 1.0
    overlay[boundary, 1] = 0.0
    overlay[boundary, 2] = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(gray_eq, cmap="gray")
    axes[0, 0].set_title("(a) Grayscale photo")
    axes[0, 1].imshow(tex, cmap="gray")
    axes[0, 1].set_title("(b) Texture map (local std)")
    axes[1, 0].imshow(tex_opened, cmap="gray")
    axes[1, 0].set_title("(c) After closing + opening")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("(d) Boundary superimposed")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "textural_segmentation.png")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.imshow(grad, cmap="gray")
    ax2.set_title("Textural segmentation gradient")
    ax2.axis("off")
    plt.tight_layout()
    save_fig(fig2, "textural_gradient.png")
    return tex, tex_opened, grad


# ------------------------------------------------------------
# 18) Main pipeline
# ------------------------------------------------------------
def main():
    rgb, gray_f, gray_eq = load_rgb_and_gray(IMAGE_PATH)

    # Save basic inputs for reference
    save_gray_image(gray_f, "input_gray.png")
    save_gray_image(gray_eq, "input_gray_enhanced.png")

    binary_mask = grabcut_foreground_mask(rgb)
    save_binary_image(binary_mask, "foreground_mask.png")

    # 1) smoothing and gradient
    op_smoothing_and_gradient(binary_mask)

    # 2) dilations
    op_binary_dilations(binary_mask)

    # 3) erosion
    op_erosion(binary_mask)

    # 4) grayscale top-hat / bottom-hat
    op_tophat_bottomhat(gray_eq)

    # 5) flat grayscale morphology
    op_gray_flat(gray_eq)

    # 6) nonflat grayscale morphology
    op_gray_nonflat(gray_eq)

    # 7) boundary extraction
    op_boundary_extraction(binary_mask, gray_eq)

    # 8) connected components
    op_connected_components(binary_mask)

    # 9) region filling
    op_region_filling(binary_mask)

    # 10) convex hull
    op_convex_hull(binary_mask)

    # 11) set operations
    op_set_operations(binary_mask)

    # 12) hit-or-miss
    op_hit_or_miss(binary_mask)

    # 13) skeletonization
    op_skeletonization(binary_mask)

    # 14) thinning
    op_thinning(binary_mask)

    # 15) thickening
    op_thickening(binary_mask)

    # 16) pruning
    op_pruning(binary_mask)

    # 17) granulometry
    op_granulometry(binary_mask)

    # 18) textural segmentation
    op_textural_segmentation(gray_eq)

    print(f"Done. All outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------
# 19) Optional: show a contact sheet in Colab/Jupyter
# ------------------------------------------------------------
# Uncomment the following block if you want to preview outputs inside the notebook.
#
# from IPython.display import Image as IPImage, display
# names = [
#     "foreground_mask.png",
#     "smooth_gradient_seed_3.png",
#     "dilation_run_7.png",
#     "dilation_run_42.png",
#     "dilation_run_2025.png",
#     "erosion_results.png",
#     "tophat_results.png",
#     "bottomhat_results.png",
#     "gray_flat_morphology.png",
#     "gray_nonflat_morphology.png",
#     "boundary_extraction.png",
#     "connected_component_result.png",
#     "cce_result.png",
#     "cce_result2.png",
#     "region_filling.png",
#     "convex_hull_example1.png",
#     "set_operations_demo.png",
#     "hitmiss_results.png",
#     "skeletonization_result.png",
#     "thinning_demo.png",
#     "thinning_result.png",
#     "thickening_demo.png",
#     "pruning_demo.png",
#     "granulometry_results.png",
#     "textural_segmentation.png",
#     "textural_gradient.png",
# ]
# for n in names:
#     p = OUT_DIR / n
#     if p.exists():
#         display(IPImage(filename=str(p)))
