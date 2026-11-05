import os
import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage
from tensorflow import keras
from scipy.ndimage import distance_transform_edt, binary_erosion
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn.keras")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


def upload_to_binary(im):
    """Uploaded image -> uint8 array: strokes = 255 (white), background = 0 (black)."""
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    rgb = np.array(im.convert("RGB"))
    gray = rgb.min(axis=2)                      # darkest channel
    if gray.mean() < 127:                       # dark background -> invert
        gray = 255 - gray
    ink = 255 - gray
    return np.where(ink > 60, 255, 0).astype("uint8")


def canvas_to_binary(data):
    """Canvas RGBA (white strokes on black) -> uint8 array, strokes = 255."""
    rgb = data[:, :, :3].astype("uint8")
    return np.where(rgb.max(axis=2) > 60, 255, 0).astype("uint8")


def segment(a):
    """Split the image into separate digits, left to right."""
    labels, _ = ndimage.label(a > 0)
    boxes = [(s[1].start, s[1].stop, s[0].start, s[0].stop)
             for s in ndimage.find_objects(labels)]
    boxes = [b for b in boxes if (b[1] - b[0]) * (b[3] - b[2]) > 150]
    boxes.sort()
    merged = []
    for b in boxes:
        if merged and b[0] < merged[-1][1]:
            m = merged[-1]
            merged[-1] = (m[0], max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3]))
        else:
            merged.append(b)
    return [a[y0:y1, x0:x1] for x0, x1, y0, y1 in merged]


def to_mnist(crop):
    """Crop (white on black) -> 28x28, strokes thinned to MNIST-like width, digit scaled to 20px and centered."""
    h, w = crop.shape
    mask = crop > 0
    half_width = distance_transform_edt(mask).max()      # about half the stroke thickness
    target = 0.07 * max(h, w)                            # MNIST-like half-thickness
    n = int(half_width - target)
    if n > 0:
        thinner = binary_erosion(mask, iterations=n)
        if thinner.sum() > 0:                            # don't erase the digit completely
            crop = (thinner * 255).astype("uint8")
            ys, xs = np.where(thinner)
            crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape
    s = 20.0 / max(h, w)
    crop = np.array(Image.fromarray(crop).resize(
        (max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))
    canvas = np.zeros((28, 28), dtype="uint8")
    y0, x0 = (28 - crop.shape[0]) // 2, (28 - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return canvas


def classify_and_show(a, model):
    """a: binary array (strokes white). Shows predicted number + per-digit results."""
    digits = segment(a)
    if not digits:
        st.warning("No digit found. Draw bigger and thicker, or use a darker image.")
        return
    canvases = [to_mnist(d) for d in digits]
    batch = np.stack(canvases).astype("float32").reshape(-1, 28, 28, 1) / 255.0
    probs = model.predict(batch, verbose=0)
    preds, confs = probs.argmax(axis=1), probs.max(axis=1)

    st.subheader("Predicted number")
    st.markdown(f"# {''.join(str(int(p)) for p in preds)}")

    st.subheader("Per-digit result")
    cols = st.columns(len(canvases))
    for col, c, p, cf in zip(cols, canvases, preds, confs):
        with col:
            st.image(Image.fromarray(c).resize((112, 112)), caption="Model input")
            st.metric("Digit", int(p), f"{cf:.0%} confident", delta_color="off")


# ---------------- UI ----------------
st.title("🔢 Handwritten Digit Recognition")
model = load_model()

tab_draw, tab_upload = st.tabs(["✏️ Draw", "📁 Upload"])

with tab_draw:
    st.write("Draw one or more digits with a gap between them, then click **Predict**.")
    stroke = st.slider("Brush size", 8, 30, 16)
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=250,
        width=600,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas",
    )
    if st.button("Predict", type="primary"):
        if canvas.image_data is None:
            st.warning("Draw something first.")
        else:
            a = canvas_to_binary(canvas.image_data)
            if a.sum() == 0:
                st.warning("Canvas is empty. Draw a digit first.")
            else:
                classify_and_show(a, model)

with tab_upload:
    st.write("Upload an image with dark digits on a white background, with a gap between digits.")
    file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    if file is not None:
        im = Image.open(file)
        st.image(im, caption="Uploaded image", width=350)
        classify_and_show(upload_to_binary(im), model)