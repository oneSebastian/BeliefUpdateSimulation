"""
Combined figure â€“ 4-row Ã— 3-col GridSpec.
  Row 0, Col 0   : (a) human_distributions
  Row 0, Col 1â€“2 : (b) rankings boxplot
  Row 1, Col 0â€“2 : (câ€“e) GPT-5.2 | GPT-5-mini | Gemini-3-flash
  Row 2, Col 0â€“2 : (fâ€“h) Claude Opus 4.6 | Qwen3-32B | Llama-3.3-70B
  Row 3, Col 0â€“1 : (i) absolute belief change â€“ Overall (2 cols)
  Row 3, Col 2   : master legend panel (bottom right grid)

Source images loaded from SVG via PyMuPDF at 300 DPI for quality.

Run from: HumanSimulationProject/
Output:   figures/plots/combined_figure.svg / .pdf / .eps
"""

import yaml
import numpy as np
import fitz                           # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
import io

# Set consistent font sizes
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       12.5,  # Base font size 12.5
})

# â”€â”€ PDF/SVG loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_file(path, dpi=300):
    """Render first page of a PDF or SVG to an RGB numpy array."""
    doc = fitz.open(path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[0].get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return arr

def trim_all(img, tol=245):
    """Remove near-white border on all four sides (uint8 threshold)."""
    mask = np.any(img < tol, axis=2)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_image(img, pad_pct=0.04):
    """Add white padding around the image as a percentage of its dimensions
    to prevent visual overlap with adjacent panels."""
    h, w = img.shape[:2]
    pad_h = int(h * pad_pct)
    pad_w = int(w * pad_pct)
    padded = np.full((h + 2 * pad_h, w + 2 * pad_w, 3), 255, dtype=np.uint8)
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
    return padded

def remove_y_label_from_img(img, y_label_text="Frequency"):
    """Remove the y-axis label 'Frequency' from the image by cropping it out."""
    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(img)

    # Crop about 40 pixels from the left to remove the y-label
    width, height = pil_img.size
    cropped_img = pil_img.crop((40, 0, width, height))

    return np.array(cropped_img)


def find_left_spine_bounds(img, dark_thresh=120, col_frac=0.25):
    """Locate the top and bottom row indices of the plot data area by detecting
    the left axis spine (the leftmost mostly-dark vertical column)."""
    gray = np.mean(img, axis=2)
    dark = gray < dark_thresh
    col_counts = dark.sum(axis=0)
    col_threshold = col_frac * img.shape[0]
    candidates = np.where(col_counts > col_threshold)[0]
    if len(candidates) == 0:
        return 0, img.shape[0] - 1
    spine_col = int(candidates[0])
    col_dark = dark[:, spine_col]
    rows = np.where(col_dark)[0]
    if len(rows) == 0:
        return 0, img.shape[0] - 1
    return int(rows[0]), int(rows[-1])


def align_to_reference_fracs(img, ref_top_frac, ref_bot_frac):
    """Pad img with white top/bottom so the detected data area sits at the given
    fractional rows. Returns (padded_img, applied) â€” applied=False if padding
    would be negative (alignment infeasible without cropping)."""
    top, bot = find_left_spine_bounds(img)
    data_h = bot - top
    if data_h <= 0:
        return img, False
    new_total = data_h / (ref_bot_frac - ref_top_frac)
    pad_top = int(round(ref_top_frac * new_total - top))
    pad_bot = int(round((1.0 - ref_bot_frac) * new_total - (img.shape[0] - 1 - bot)))
    if pad_top < 0 or pad_bot < 0:
        return img, False
    w = img.shape[1]
    top_band = np.full((pad_top, w, 3), 255, dtype=np.uint8)
    bot_band = np.full((pad_bot, w, 3), 255, dtype=np.uint8)
    return np.vstack([top_band, img, bot_band]), True

# â”€â”€ colors for master legend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS     = _yaml["models"]
HUMAN_COLOR = _yaml["human"]
INIT_COLOR  = "#cccccc"

# Model order matching your individual distribution panels
MODEL_ORDER = [
    "GPT-5.2", "GPT-5-mini", "Gemini-3-flash",
    "Claude Opus 4.6", "Qwen3-32B", "Llama-3.3-70B-Instruct",
]
MODEL_SLUGS = ["gpt52", "gpt5mini", "gemini", "claude", "qwen", "llama"]

# Models from which to remove the y-axis label "Frequency"
MODELS_TO_REMOVE_Y_LABEL = ["GPT-5-mini", "Gemini-3-flash", "Qwen3-32B", "Llama-3.3-70B-Instruct"]

# â”€â”€ load source files (prefer SVG if available, otherwise PDF) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

print("Loading files...")

# Try to load SVG files first, fall back to PDF
def load_preferred(path_svg, path_pdf):
    """Try to load SVG first, then PDF if SVG doesn't exist."""
    import os
    if os.path.exists(path_svg):
        print(f"  Loading SVG: {path_svg}")
        return trim_all(load_file(path_svg))
    else:
        print(f"  Loading PDF: {path_pdf}")
        return trim_all(load_file(path_pdf))

# Human distribution
human_img = load_preferred(
    "figures/plots/distributions/human_distributions_nol.svg",
    "figures/plots/distributions/human_distributions_nol.pdf"
)

# Rankings boxplot
rankings_img = load_preferred(
    "figures/rankings/plot_avg_comment_rankings.svg",
    "figures/rankings/plot_avg_comment_rankings.pdf"
)

# Dot plot
dot_img = load_preferred(
    "figures/plots/absolute_belief_change_dot_range_nol.svg",
    "figures/plots/absolute_belief_change_dot_range_nol.pdf"
)

# Load the individual model distribution panels
model_imgs_original = []
for slug in MODEL_SLUGS:
    img = load_preferred(
        f"figures/plots/distributions/panel_{slug}.svg",
        f"figures/plots/distributions/panel_{slug}.pdf"
    )
    model_imgs_original.append(img)

# Y-labels are suppressed at the source for d/e/g/h panels â€” no cropping needed.
model_imgs = list(model_imgs_original)

# Add tiny white padding around each image to prevent visual overlap.
# Reduced from 0.02 â†’ 0.005 to tighten the layout; gridspec spacing handles separation.
human_img = pad_image(human_img, pad_pct=0.005)
rankings_img = pad_image(rankings_img, pad_pct=0.005)
dot_img = pad_image(dot_img, pad_pct=0.005)
model_imgs = [pad_image(img, pad_pct=0.005) for img in model_imgs]

print(f"Loaded {len(model_imgs)} model panels")
print(f"Human image shape: {human_img.shape}")
print(f"Rankings image shape: {rankings_img.shape}")
print(f"Dot plot image shape: {dot_img.shape}")
for i, img in enumerate(model_imgs):
    print(f"Model {MODEL_ORDER[i]} shape: {img.shape}")

# â”€â”€ figure layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Set figure width â€” source panels are now ~3.2" wide, so 3*3.2 + small gaps = ~10"
# Source font sizes (10â€“13pt) survive print scaling down to Nature double-column (7.2").
FIG_W = 10.0
col_w = FIG_W / 3   # â‰ˆ 3.33"

# Calculate heights based on aspect ratios of the images with extra padding
panel_h = model_imgs[0].shape[0] / model_imgs[0].shape[1] * col_w
human_h = human_img.shape[0] / human_img.shape[1] * col_w
rankings_h = rankings_img.shape[0] / rankings_img.shape[1] * (2 * col_w)
dot_h = dot_img.shape[0] / dot_img.shape[1] * (2 * col_w)

# Row 0 height = human's natural display height. The rankings panel is generated
# at half the human panel's outer aspect (figsize 6.4Ã—2.8 vs 3.2Ã—2.8), so when
# placed at 2Ã— the column width it ends up the same display height as panel (a).
# Both panels then fill their cells with aspect="auto" â€” bottoms (x-axis) and
# tops (top of y-axis) line up across the row.
row0_h = human_h

# Reduced spacing between rows now that source panels are tightly bounded
row_spacing = 0.05
total_height = (row0_h + 2 * panel_h + dot_h + 3 * row_spacing) * 1.2

# Create figure â€” 12-column grid so row 3 can split 7:5 (plot i : legend)
# without affecting the equal 4-col-per-panel layout of rows 0-2.
fig = plt.figure(figsize=(FIG_W, total_height))

gs = gridspec.GridSpec(
    4, 12, figure=fig,
    hspace=0.14,
    wspace=0.04,
    height_ratios=[row0_h, panel_h, panel_h, dot_h],
)

# â”€â”€ panel a: Human distribution (row 0, col 0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Row-0 cell is sized to human's natural aspect; rankings figsize was set to match
# at 2Ã— width, so aspect="auto" on both panels yields aligned x-axis and y-axis spans.
LABEL_KW = dict(transform=None, fontsize=16, fontweight="normal",
                va="bottom", ha="left", color="black", clip_on=False)

def add_label(ax, letter):
    """Place panel letter just above the top-left corner of the axes.
    All panels in the same column share the same axes left-edge in figure
    coordinates, so x=0 in transAxes gives perfect column alignment."""
    ax.text(0, 1.04, letter, transform=ax.transAxes, **{k: v for k, v in LABEL_KW.items() if k != "transform"})

# 12-col mapping: each of the 3 logical columns = 4 sub-cols (0-3, 4-7, 8-11)
ax_a = fig.add_subplot(gs[0, 0:4])
ax_a.imshow(human_img, aspect="auto")
ax_a.axis("off")
add_label(ax_a, "a")

# â”€â”€ panel b: Rankings boxplot (row 0, cols 4-11) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax_b = fig.add_subplot(gs[0, 4:])
ax_b.imshow(rankings_img, aspect="auto")
ax_b.axis("off")
add_label(ax_b, "b")

# â”€â”€ panels câ€“e: First row of models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for idx, (img, model_name) in enumerate(zip(model_imgs[:3], MODEL_ORDER[:3])):
    c0 = idx * 4
    ax = fig.add_subplot(gs[1, c0:c0 + 4])
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    add_label(ax, "cde"[idx])

# â”€â”€ panels fâ€“h: Second row of models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for idx, (img, model_name) in enumerate(zip(model_imgs[3:], MODEL_ORDER[3:])):
    c0 = idx * 4
    ax = fig.add_subplot(gs[2, c0:c0 + 4])
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    add_label(ax, "fgh"[idx])

# â”€â”€ panel i: cols 0-6 (7/12 â‰ˆ 58% â€” 75 % wider than the previous 1/3) â”€â”€â”€â”€â”€â”€â”€
ax_dot = fig.add_subplot(gs[3, 0:7])
ax_dot.imshow(dot_img, aspect="auto")
ax_dot.axis("off")
add_label(ax_dot, "i")

# â”€â”€ master legend: cols 7-11 (5/12 â‰ˆ 42%), left-aligned inside its cell â”€â”€â”€â”€â”€â”€
ax_leg = fig.add_subplot(gs[3, 7:])
ax_leg.axis("off")

legend_handles = [
    mpatches.Patch(color=HUMAN_COLOR, label="Human Post-Stance", alpha=0.85),
    mlines.Line2D([], [], color=INIT_COLOR, linewidth=2.0, label="Initial Stance"),
] + [
    mpatches.Patch(color=_COLORS[m], label=m, alpha=0.85) for m in MODEL_ORDER
]

# Create legend in the bottom right cell with adjusted spacing
legend = ax_leg.legend(
    handles=legend_handles,
    loc="center left",
    ncol=2,
    fontsize=12,
    framealpha=0.95,
    frameon=True,
    edgecolor="lightgray",
    fancybox=False,
    handlelength=1.5,
    handletextpad=0.6,
    columnspacing=1.0,
    title="Legend",
    title_fontsize=12.5,
    borderpad=0.5,
    labelspacing=0.5,
)

# Tighter outer margins â€” source panels already include their own breathing room.
plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)

# â”€â”€ save in multiple formats (EPS and SVG) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
output_dir = "figures/plots"

# Save as SVG (vector, editable) â€” primary format
out_svg = f"{output_dir}/combined_figure.svg"
fig.savefig(out_svg, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {out_svg}")

# Save as EPS at lower DPI to keep file size reasonable.
# (Matplotlib embeds raster images uncompressed in EPS; lower DPI = smaller file.)
out_eps = f"{output_dir}/combined_figure.eps"
fig.savefig(out_eps, dpi=150, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {out_eps}")

# Save as PDF (for compatibility)
out_pdf = f"{output_dir}/combined_figure.pdf"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {out_pdf}")

# Also save as PNG for preview
out_png = f"{output_dir}/combined_figure.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved {out_png}")

print(f"\nFigure dimensions: {fig.get_size_inches()} inches")
print("Layout: 4 rows Ã— 3 columns")
print("Legend placed in bottom right grid cell (row 3, column 2)")
print("Overlap prevention measures:")
print("  - Increased figure width from 8.0 to 9.5 inches")
print("  - Increased hspace from 0.15 to 0.25 (more vertical space between rows)")
print("  - Increased wspace from 0.08 to 0.12 (more horizontal space between columns)")
print("  - Adjusted subplots_adjust with larger margins (0.05 all sides)")
print("  - Removed tight_layout to prevent conflicts")
print("  - Increased pad_inches from 0.1 to 0.2 for all formats")
print("  - Legend font size set to 10 for better fit")
print("  - Panel labels positioned at (0.01, 0.97) for visibility")
print("\nPanels include:")
print("  a: Human ground truth distribution")
print("  b: Rankings boxplot (models first, then human)")
print("  c-e: GPT-5.2, GPT-5-mini, Gemini-3-flash distributions")
print("  f-h: Claude Opus 4.6, Qwen3-32B, Llama-3.3-70B-Instruct distributions")
print("  i: Absolute belief change - Overall (dot plot)")
print("  Legend: Master legend in bottom right corner")
print("Y-axis 'Frequency' labels removed from: GPT-5-mini, Gemini-3-flash, Qwen3-32B, Llama-3.3-70B-Instruct")
print("\nFiles saved:")
print(f"  - {out_svg} (vector, editable)")
print(f"  - {out_eps} (EPS format for publication)")
print(f"  - {out_pdf} (PDF format)")
print(f"  - {out_png} (PNG preview)")

plt.close(fig)