"""
Merge zscore_plot_log_scale.svg (left, panel a) with
ablation_stances_extended.svg (right, panel b).

Full two-column width for Nature Communications (180 mm = 510 pt).
Both panels scaled to the same height, placed side by side.

Run from: HumanSimulationProject/
Output:   figures/plots/combined_zscore_ablation.svg / .png / .eps
"""

import fitz                              # PyMuPDF â€“ for PNG/EPS rasterisation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from svgutils.transform import SVGFigure, fromfile
import svgutils.transform as sg
import os

ZSCORE_PATH   = "figures/plots/zscore_plots/zscore_plot_log_scale.svg"
ABLATION_PATH = "figures/plots/distributions/ablation_stances_extended.svg"
OUT_DIR       = "figures/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# â”€â”€ get native SVG dimensions via PyMuPDF (in points) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def svg_dims(path):
    doc = fitz.open(path)
    r = doc[0].rect
    doc.close()
    return r.width, r.height

W_A, H_A = svg_dims(ZSCORE_PATH)
W_B, H_B = svg_dims(ABLATION_PATH)
print(f"zscore:   {W_A:.1f} Ã— {H_A:.1f} pt")
print(f"ablation: {W_B:.1f} Ã— {H_B:.1f} pt")

# â”€â”€ target layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Nature Communications full two-column width = 180 mm = 510.24 pt
NC_FULL_W = 510.24
GAP       = 12.0   # gap between panels in pt

panel_w   = (NC_FULL_W - GAP) / 2.0

scale_a   = panel_w / W_A
scale_b   = panel_w / W_B

h_a_sc    = H_A * scale_a
h_b_sc    = H_B * scale_b
fig_h     = max(h_a_sc, h_b_sc)

print(f"Panel width: {panel_w:.1f} pt  |  Figure height: {fig_h:.1f} pt")

# â”€â”€ build combined vector SVG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
zscore_fig   = fromfile(ZSCORE_PATH)
ablation_fig = fromfile(ABLATION_PATH)

root_a = zscore_fig.getroot()
root_b = ablation_fig.getroot()

root_a.moveto(0, 0, scale_a)
root_b.moveto(panel_w + GAP, 0, scale_b)

LABEL_SIZE = 16
label_a = sg.TextElement(4, LABEL_SIZE + 2, "a",
                          size=LABEL_SIZE, weight="bold", font="Arial")
label_b = sg.TextElement(panel_w + GAP + 4, LABEL_SIZE + 2, "b",
                          size=LABEL_SIZE, weight="bold", font="Arial")

combined = SVGFigure()
combined.set_size((f"{NC_FULL_W}pt", f"{fig_h}pt"))
combined.append([root_a, root_b, label_a, label_b])

svg_out = f"{OUT_DIR}/combined_zscore_ablation.svg"
combined.save(svg_out)
print(f"Saved {svg_out}")

# â”€â”€ rasterise combined SVG â†’ PNG and EPS via matplotlib â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_svg_as_array(path, dpi=300):
    doc = fitz.open(path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[0].get_pixmap(matrix=mat, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    return arr

combined_arr = load_svg_as_array(svg_out, dpi=300)

# figure size in inches: convert pt â†’ inches (72 pt/inch)
fig_w_in = NC_FULL_W / 72
fig_h_in = fig_h / 72

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
ax.imshow(combined_arr, aspect="auto")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

for fmt in ("png", "eps"):
    out = f"{OUT_DIR}/combined_zscore_ablation.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
    print(f"Saved {out}")

plt.close(fig)
