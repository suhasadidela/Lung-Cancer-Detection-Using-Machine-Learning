import gradio as gr
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog
import joblib
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============ Load models ============
cnn_model = load_model("cnn_model.h5")      # or cnn_model.keras
xgb_model = joblib.load("xgb_model.pkl")

# ============ Preprocessing ============
def preprocess_image(image):
    if image is None:
        raise ValueError("No image provided.")
    # convert to grayscale if needed
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image_resized = transform.resize(image, (100, 100))
    image_norm = image_resized / 255.0
    return image_norm

# ============ Prediction ============
def predict_cancer(image):
    try:
        image = preprocess_image(image)

        # CNN probabilities
        cnn_input = image.reshape(1, 100, 100, 1)
        cnn_probs = cnn_model.predict(cnn_input, verbose=0)

        # XGBoost probabilities
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L2-Hys"
        )
        xgb_probs = xgb_model.predict_proba([features])

        # Bayesian weighted average
        cnn_acc, xgb_acc = 0.98, 0.97
        w_cnn = cnn_acc / (cnn_acc + xgb_acc)
        w_xgb = xgb_acc / (cnn_acc + xgb_acc)
        final_probs = w_cnn * cnn_probs + w_xgb * xgb_probs

        cancer_prob = float(final_probs[0][1]) * 100  # class 1 = malignant
        return f"{cancer_prob:.2f}%"
    except Exception as e:
        return f"Error: {e}"

# ============ Gradio Interface ============
interface = gr.Interface(
    fn=predict_cancer,
    inputs=gr.Image(type="numpy", label="Upload CT Scan"),
    outputs=gr.Textbox(label="Cancer Probability (%)"),
    title="Lung Cancer Detection",
    description="Upload a CT scan image to estimate the probability of lung cancer."
)

if __name__ == "__main__":
    interface.launch()
