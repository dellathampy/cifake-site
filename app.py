"""
CIFAKE: Real vs AI-Generated Image Classifier — Web Demo
"""

import os
import numpy as np
import gradio as gr
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm
from PIL import Image

MODEL_PATH = "cifake_model.keras"
IMG_SIZE = 32

# ---------- Load model once at startup ----------
model = tf.keras.models.load_model(MODEL_PATH)

# Find the last Conv2D layer automatically (same trick as the notebook)
last_conv_layer_name = [
    layer.name for layer in model.layers
    if isinstance(layer, tf.keras.layers.Conv2D)
][-1]

# Rebuild the model as an explicit functional graph using a symbolic
# Input, tracking the last conv layer's output along the way. This is
# necessary because a Sequential model loaded fresh from disk doesn't
# have `.output` defined until it's used inside a functional graph —
# calling it on real (eager) data does NOT count, only symbolic does.
_inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
_x = _inputs
_conv_output = None
for layer in model.layers:
    _x = layer(_x)
    if layer.name == last_conv_layer_name:
        _conv_output = _x

grad_model = tf.keras.models.Model(inputs=_inputs, outputs=[_conv_output, _x])


def make_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # single sigmoid output

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (original_img.shape[1], original_img.shape[0])
        )
    )
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    overlay = np.uint8(jet_heatmap * alpha + original_img * (1 - alpha))
    return overlay


def predict(image: Image.Image):
    if image is None:
        return None, "Please upload an image."

    image = image.convert("RGB")
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(resized).astype("float32")
    img_batch = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_batch, verbose=0)[0][0]
    label = "FAKE (AI-generated)" if pred > 0.5 else "REAL"
    confidence = pred if pred > 0.5 else 1 - pred

    heatmap = make_gradcam_heatmap(img_batch)
    overlay = overlay_heatmap(img_array, heatmap)

    result_text = f"**{label}**\n\nConfidence: {confidence * 100:.1f}%"
    return Image.fromarray(overlay), result_text


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM (where the model looked)"),
        gr.Markdown(label="Prediction"),
    ],
    title="CIFAKE: Real vs AI-Generated Image Classifier",
    description=(
        "Upload an image and the model will predict whether it's a REAL "
        "photograph or an AI-GENERATED (fake) image, based on the CIFAKE "
        "dataset. The heatmap shows which regions most influenced the "
        "decision (Grad-CAM)."
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
