# CIFAKE: Real vs AI-Generated Image Classifier

A web app that tells you whether an image is a real photograph or AI-generated, built on the CIFAKE dataset. Upload an image and it predicts REAL or FAKE, along with a Grad-CAM heatmap showing which part of the image the model based its decision on.

**Live demo:** https://cifake-site.onrender.com

## What this is

This started as a notebook-based project comparing real photos against AI-generated images (from the CIFAKE dataset on Kaggle: `birdy654/cifake-real-and-ai-generated-synthetic-images`). Once the model was trained, I wrapped it in a small Gradio app so it could actually be used from a browser instead of just sitting in a notebook.

## How the model works

The classifier is a small CNN, matching the best-performing architecture from the original CIFAKE paper:

- 2 convolutional layers (32 filters each) with max pooling
- A dense layer of 64 units
- A single sigmoid output for binary classification (real vs fake)

Images are 32x32 RGB, trained on roughly 20,000 images (10,000 per class) out of the full 100,000+ image dataset — enough to get a solid result without the training time of the full set.

**Results on the test set (6,400 images):**
- Accuracy: 90%
- FAKE — precision 0.88, recall 0.92, F1 0.90
- REAL — precision 0.92, recall 0.88, F1 0.90
- Weighted avg F1: 0.90
- Compared against the base paper's reported 93.55% accuracy / 0.936 F1 — close, given this model trained on a ~20,000-image subset rather than the full 100,000+ dataset

(Full training curves and confusion matrix are in the notebook, `CIFAKE_Project_Notebook.ipynb`.)

## Grad-CAM

The interesting part isn't just the prediction, it's seeing *why*. Grad-CAM highlights the pixels that most influenced the model's decision by looking at gradients flowing back into the last convolutional layer. In practice, real images tend to light up more evenly across the frame, while fake images tend to show the model fixating on smaller, weirder regions — textures or artifacts that don't look "natural." That pattern shows up on the demo site as a heatmap overlay next to each prediction.

## Tech stack

- **Model:** TensorFlow / Keras (CNN)
- **Web app:** Gradio
- **Hosting:** Render (free tier)

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Make sure `cifake_model.keras` is in the same folder as `app.py` before running.

## Project structure

```
app.py                 - the web app (loads the model, runs predictions + Grad-CAM)
requirements.txt        - dependencies
cifake_model.keras      - trained model
CIFAKE_Project_Notebook.ipynb - training notebook (data loading, training, evaluation)
```
