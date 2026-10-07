"""
predict.py  (version 5)
- Any number of digits (3, 4, 6, ...)
- Two ways to split digits:
    "auto"  : connected components + model-checked splitting of touching digits
    "slots" : fixed guide boxes (one digit per box) - most reliable
- Stroke-thickness normalization + MNIST-style centering
- Ensemble: averages the predictions of all models in the models/ folder

Test from terminal:  python predict.py my_image.png
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

MODELS_DIR = "models"


def load_models(folder=MODELS_DIR):
    from tensorflow import keras
    paths = sorted(glob.glob(os.path.join(folder, "*.keras")))
    if not paths and os.path.exists("digit_model.keras"):
        paths = ["digit_model.keras"]
    if not paths:
        raise FileNotFoundError("No trained model found. Run train_model.py first.")
    return [keras.models.load_model(p) for p in paths]


# ------------------------------------------------------------
# Step 1: white-digit-on-black, clean background
# ------------------------------------------------------------
def to_white_on_black(img, auto_invert=True):
    arr = np.array(img.convert("L")).astype("float32")

    if auto_invert and arr.mean() > 127:      # black ink on white paper -> flip
        arr = 255 - arr

    if arr.max() > arr.min():                 # stretch contrast to 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    arr[arr < 60] = 0                         # remove faint background noise
    return arr.astype("uint8")


def _crop(piece):
    ys, xs = np.where(piece > 0)
    if len(xs) == 0:
        return None
    return piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ------------------------------------------------------------
# Step 2a: decide if a wide blob is really several touching digits.
# Width alone is NOT reliable (a "2" with a long base is wide but is ONE digit).
# So we use the model to help ("segmentation by recognition"):
#   1. find possible cut points (columns with little ink)
#   2. ask the model how sure it is about every possible piece
#   3. pick the cuts that give the most confident digits overall
# ------------------------------------------------------------
WIDE_RATIO = 0.85        # blobs wider than this (width/height) get checked
PIECE_PENALTY = 0.08     # each extra piece must raise average confidence by this much
MIN_PART_W = 10          # a piece must be at least this wide (pixels)
MIN_PART_H_FRAC = 0.5    # ...at least half as tall as the blob (rejects flat bars)
MAX_PART_RATIO = 1.5     # ...and not absurdly wide
MAX_CUTS = 24            # keep the search fast
MAX_PIECES = 8


def classify(models, digit_imgs):
    """28x28 images -> probability array (n, 10), averaged over all models."""
    batch = np.array(digit_imgs).reshape(-1, 28, 28, 1)
    return np.mean([m.predict(batch, verbose=0) for m in models], axis=0)


def candidate_cuts(piece):
    """Local minima of the ink-per-column curve = places where digits may join."""
    h, w = piece.shape
    col = (piece > 0).sum(axis=0).astype("float32")
    smooth = ndimage.uniform_filter1d(col, size=5)
    peaks, _ = find_peaks(-smooth, distance=max(MIN_PART_W, int(0.25 * h)))
    peaks = [int(p) for p in peaks if MIN_PART_W <= p <= w - MIN_PART_W]
    return sorted(sorted(peaks, key=lambda p: smooth[p])[:MAX_CUTS])


def _valid_part(part, blob_h):
    if part is None:
        return False
    ph, pw = part.shape
    return pw >= MIN_PART_W and ph >= MIN_PART_H_FRAC * blob_h and pw / ph <= MAX_PART_RATIO


def resolve_blob(models, piece):
    """Returns a list of 1 or more digit pieces for this blob."""
    h, w = piece.shape
    if w / h <= WIDE_RATIO:
        return [piece]

    cuts = [0] + candidate_cuts(piece) + [w]
    n = len(cuts)

    # every possible piece between two cut points
    keys, parts, imgs = [], [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            part = _crop(piece[:, cuts[i]:cuts[j]])
            is_whole = (i == 0 and j == n - 1)          # "one single digit" is always allowed
            if part is not None and (is_whole or _valid_part(part, h)):
                keys.append((i, j))
                parts.append(part)
                imgs.append(to_mnist_28(part))
    if len(imgs) <= 1:
        return [piece]

    probs = classify(models, imgs)                       # one batch for all pieces
    conf = {k: float(p.max()) for k, p in zip(keys, probs)}
    part_of = dict(zip(keys, parts))

    # dynamic programming: best total confidence using exactly k pieces up to cut j
    NEG = -1e9
    kmax = min(MAX_PIECES, n - 1)
    best = [[NEG] * (kmax + 1) for _ in range(n)]
    back = {}
    best[0][0] = 0.0
    for j in range(1, n):
        for i in range(j):
            if (i, j) not in conf:
                continue
            for k in range(1, kmax + 1):
                if best[i][k - 1] > NEG / 2:
                    v = best[i][k - 1] + conf[(i, j)]
                    if v > best[j][k]:
                        best[j][k] = v
                        back[(j, k)] = i

    best_k, best_score = None, NEG
    for k in range(1, kmax + 1):
        if best[n - 1][k] > NEG / 2:
            score = best[n - 1][k] / k - PIECE_PENALTY * (k - 1)
            if score > best_score:
                best_score, best_k = score, k
    if best_k is None:
        return [piece]

    segs, j, k = [], n - 1, best_k
    while k > 0:
        i = back[(j, k)]
        segs.append((i, j))
        j, k = i, k - 1
    segs.reverse()
    return [part_of[s] for s in segs]


# ------------------------------------------------------------
# Step 2b: connected components ("auto" mode)
# ------------------------------------------------------------
def segment_components(arr):
    mask = arr > 0
    if not mask.any():
        return []

    eight = np.ones((3, 3), dtype=int)        # 8-connectivity
    dil = ndimage.binary_dilation(mask, structure=eight, iterations=1)
    labels, _ = ndimage.label(dil, structure=eight)
    labels = labels * mask                    # keep only the real ink

    comps = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        comps.append({"ids": [i], "x1": xs.start, "x2": xs.stop - 1,
                      "y1": ys.start, "y2": ys.stop - 1,
                      "area": int((labels[sl] == i).sum())})
    if not comps:
        return []

    max_area = max(c["area"] for c in comps)
    comps = [c for c in comps if c["area"] >= 0.03 * max_area]   # drop specks
    max_h = max(c["y2"] - c["y1"] + 1 for c in comps)

    big = [c for c in comps if (c["y2"] - c["y1"] + 1) >= 0.4 * max_h]
    small = [c for c in comps if c not in big]

    # attach small pieces (e.g. bar of a "5") to the big digit above/below them
    for s in small:
        sw = s["x2"] - s["x1"] + 1
        best, best_ov = None, 0
        for b in big:
            ov = min(s["x2"], b["x2"]) - max(s["x1"], b["x1"]) + 1
            if ov > best_ov:
                best, best_ov = b, ov
        if best is not None and best_ov >= 0.5 * sw:
            best["ids"] += s["ids"]
            best["x1"], best["x2"] = min(best["x1"], s["x1"]), max(best["x2"], s["x2"])
            best["y1"], best["y2"] = min(best["y1"], s["y1"]), max(best["y2"], s["y2"])

    big.sort(key=lambda c: c["x1"])           # left to right

    digits = []
    for b in big:
        m = np.isin(labels, b["ids"])
        digits.append(_crop(arr * m))
    return digits


# ------------------------------------------------------------
# Step 2c: fixed boxes ("slots" mode)
# ------------------------------------------------------------
def segment_slots(arr, n_slots):
    h, w = arr.shape
    slot_w = w // n_slots
    digits = []
    for i in range(n_slots):
        piece = arr[:, i * slot_w:(i + 1) * slot_w]
        if (piece > 0).sum() < 30:            # empty box (or just a speck)
            continue
        digits.append(_crop(piece))
    return digits


# ------------------------------------------------------------
# Step 3: one cropped digit -> 28x28 MNIST-style image
# ------------------------------------------------------------
def normalize_stroke(piece, scale):
    """
    MNIST strokes are ~2-3 px wide after resizing. If our pen is thinner than that
    (after scaling down), thicken it so the digit looks like MNIST.
    """
    mask = np.pad(piece > 0, 1)
    dt = ndimage.distance_transform_edt(mask)[1:-1, 1:-1]
    vals = dt[piece > 0]
    if len(vals) == 0:
        return piece
    stroke = 2 * np.percentile(vals, 90)      # estimated stroke width (pixels)
    target = 2.6 / scale                      # wanted width before shrinking
    if stroke < 0.8 * target:
        k = int(round((target - stroke) / 2))
        if k > 0:
            piece = np.pad(piece, k)
            piece = ndimage.grey_dilation(piece, size=(2 * k + 1, 2 * k + 1))
    return piece


def to_mnist_28(digit):
    h, w = digit.shape
    digit = normalize_stroke(digit, 20.0 / max(h, w))
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = Image.fromarray(digit).resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - new_w) // 2, (28 - new_h) // 2))

    # shift so the centre of mass is in the middle (this is how MNIST was made)
    a = np.array(canvas).astype("float32")
    total = a.sum()
    if total > 0:
        ys, xs = np.indices(a.shape)
        cy = (ys * a).sum() / total
        cx = (xs * a).sum() / total
        dx, dy = 14 - cx, 14 - cy
        canvas = canvas.transform((28, 28), Image.AFFINE,
                                  (1, 0, -dx, 0, 1, -dy), resample=Image.BILINEAR)

    return np.array(canvas).astype("float32") / 255.0


def recognize(models, img, auto_invert=True, mode="auto", n_slots=8):
    """
    Full pipeline: image -> number.
    Returns (number_string, details, digit_images)
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    arr = to_white_on_black(img, auto_invert)

    if mode == "slots":
        pieces = segment_slots(arr, n_slots)
    else:
        pieces = []
        for blob in segment_components(arr):
            pieces += resolve_blob(models, blob)

    digit_imgs = [to_mnist_28(p) for p in pieces]
    number, details = predict_number(models, digit_imgs)
    return number, details, digit_imgs


# ------------------------------------------------------------
# Step 4: prediction (average of all models)
# ------------------------------------------------------------
def predict_number(models, digit_images):
    """Returns (number_string, details); details = [(digit, conf%, top3), ...]"""
    if not digit_images:
        return "", []
    if not isinstance(models, (list, tuple)):
        models = [models]

    batch = np.array(digit_images).reshape(-1, 28, 28, 1)
    probs = np.mean([m.predict(batch, verbose=0) for m in models], axis=0)

    details = []
    for p in probs:
        top3_idx = np.argsort(p)[::-1][:3]
        top3 = [(int(i), float(p[i] * 100)) for i in top3_idx]
        details.append((top3[0][0], top3[0][1], top3))

    return "".join(str(d[0]) for d in details), details


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    models = load_models()
    number, details, _ = recognize(models, Image.open(sys.argv[1]))
    if not number:
        print("No digit found.")
    else:
        print("Predicted number:", number)
        for i, (d, c, t) in enumerate(details, 1):
            print(f"  digit {i}: {d} ({c:.1f}%)  top3={t}")