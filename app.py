import numpy as np, gradio as gr
from PIL import Image
from scipy import ndimage
from tensorflow import keras

model = keras.models.load_model("mnist_cnn.keras")


def to_binary(img):
    """Return a uint8 array: strokes = 255 (white), background = 0 (black)."""
    if img is None:
        return None
    im = Image.fromarray(np.array(img).astype("uint8"))
    if im.mode == "RGBA":                          # flatten transparency onto white
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    rgb = np.array(im.convert("RGB"))
    gray = rgb.min(axis=2)                         # darkest channel: works for colored strokes
    if gray.mean() < 127:                          # dark background -> invert
        gray = 255 - gray
    ink = 255 - gray                               # strokes become bright
    return np.where(ink > 60, 255, 0).astype("uint8")


def segment(a):
    """Split the image into separate digits, left to right."""
    labels, _ = ndimage.label(a > 0)
    boxes = [(s[1].start, s[1].stop, s[0].start, s[0].stop)
             for s in ndimage.find_objects(labels)]
    boxes = [b for b in boxes if (b[1] - b[0]) * (b[3] - b[2]) > 150]   # drop specks
    boxes.sort()
    merged = []
    for b in boxes:                                # join pieces of one digit (e.g. 5, 4)
        if merged and b[0] < merged[-1][1]:
            m = merged[-1]
            merged[-1] = (m[0], max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3]))
        else:
            merged.append(b)
    return [a[y0:y1, x0:x1] for x0, x1, y0, y1 in merged]


def to_mnist(crop):
    """Crop (white on black) -> 28x28, digit scaled to 20px and centered."""
    h, w = crop.shape
    s = 20.0 / max(h, w)
    crop = np.array(Image.fromarray(crop).resize(
        (max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS))
    canvas = np.zeros((28, 28), dtype="uint8")
    y0, x0 = (28 - crop.shape[0]) // 2, (28 - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    Image.fromarray(canvas).resize((280, 280)).save("debug.png")   # what the model sees
    return canvas.astype("float32").reshape(1, 28, 28, 1) / 255.0


def predict(img):
    a = to_binary(img)
    if a is None:
        return "Please upload an image"
    digits = segment(a)
    if not digits:
        return "No digit found"
    results = []
    for d in digits:
        p = model.predict(to_mnist(d), verbose=0)[0]
        results.append((int(p.argmax()), float(p.max())))
    number = "".join(str(d) for d, _ in results)
    conf = ", ".join(f"{d}: {c:.0%}" for d, c in results)
    return f"Predicted number: {number}\nPer-digit confidence -> {conf}"


gr.Interface(
    predict,
    gr.Image(type="numpy", image_mode="RGB", sources=["upload", "clipboard"],
             label="Upload digit image"),
    gr.Textbox(label="Prediction"),
    title="Handwritten Digit Recognition",
).launch()