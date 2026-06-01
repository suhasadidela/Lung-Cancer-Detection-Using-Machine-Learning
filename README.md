# 🫁 Lung Cancer Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green)
![Gradio](https://img.shields.io/badge/Gradio-App-purple)
![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-99.52%25-brightgreen)

> **Published Undergraduate Capstone Project** — *"A Stacking Ensemble-based Approach for Lung Cancer Detection"*  
> SRM Institute of Science and Technology, Department of Computational Intelligence

🚀 **[Live Demo on Hugging Face](https://huggingface.co/spaces/suhasadidela/Lung_Cancer_detection)** — Upload a CT scan and get an instant cancer probability estimate!

---

## 📌 Overview

Early detection of lung cancer dramatically improves patient survival rates. Traditional diagnostic methods are often invasive, expensive, and variable in accuracy. This project introduces a **stacking ensemble methodology** applied to CT scan images, combining deep learning and classical ML models using **logistic regression as a meta-model** to produce a highly accurate, non-invasive diagnostic tool.

**My Role:** Data collection & preprocessing

---

## 📊 Results

### Individual Model Performance on IQ-OTH/NCCD Dataset

| Model | Accuracy | Precision | F1 Score | Recall |
|---|---|---|---|---|
| SVM ⭐ | 97.59% | 0.94 | 0.96 | 0.98 |
| Logistic Regression (meta) | 96.87% | 0.98 | 0.99 | 1.00 |
| Random Forest ⭐ | 93.98% | 0.96 | 0.97 | 0.99 |
| KNN | 97.11% | 0.99 | 0.99 | 0.99 |
| Naive Bayes | 62.05% | 0.83 | 0.78 | 0.75 |
| Decision Tree | 89.88% | 0.95 | 0.95 | 0.95 |
| XGBoost ⭐ | 97.35% | 0.98 | 0.99 | 1.00 |
| AdaBoost | 78.55% | 0.90 | 0.90 | 0.84 |
| CNN (100 epochs) ⭐ | 97.83% | 1.00 | 0.99 | 0.98 |
| RBM (100 epochs) | 93.73% | 0.96 | 0.97 | 0.98 |
| **Stacking Ensemble** | **99.52%** | **0.9911** | — | **0.9907** |

> ⭐ = used as base model in stacking ensemble | ROC-AUC = **1.00** across all classes

### Key Takeaway
The stacking ensemble (CNN + XGBoost + Random Forest + SVM, with Logistic Regression as meta-model) **outperforms every individual model**, achieving 99.52% accuracy and a perfect AUC of 1.00.

---

## 🧠 How It Works

1. **Input**: CT scan image uploaded by user
2. **Preprocessing**: Grayscale → resize to 100×100 → normalize to [0,1]
3. **Base Models**: CNN, XGBoost, Random Forest, SVM each make predictions
4. **Meta-Model**: Logistic Regression combines base model outputs
5. **Output**: Final classification — Benign / Malignant / Normal

The deployed Hugging Face app uses a **Bayesian weighted ensemble of CNN + XGBoost** specifically, for fast real-time inference.

---

## 📁 Dataset

- **Name**: IQ-OTH/NCCD Lung Cancer Dataset
- **Source**: Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases
- **Classes**: Benign, Malignant, Normal
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras (CNN) |
| ML Libraries | scikit-learn, XGBoost |
| Image Processing | scikit-image (HOG features) |
| Web App | Gradio |
| Training Environment | Google Colab |
| Deployment | Hugging Face Spaces |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/suhasadidela/Lung-Cancer-Detection-Using-Machine-Learning.git
cd Lung-Cancer-Detection-Using-Machine-Learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model files from Hugging Face (too large for GitHub)
# https://huggingface.co/spaces/suhasadidela/Lung_Cancer_detection/tree/main
# Download: cnn_model.h5 and xgb_model.pkl → place in same folder

# 4. Run the app
python app.py
```

> ⚠️ `cnn_model.h5` (52MB) is hosted on Hugging Face due to GitHub's file size limit.

---

## 📁 Repository Structure

```
├── LungCancerDetectionUsingMachine_Learning.ipynb  # Full training notebook (10+ models)
├── app.py                                           # Gradio web app
├── requirements.txt                                 # Python dependencies
├── splnproc1703 (3).docm                           # Published research paper
└── README.md
```

> Model files (`cnn_model.h5`, `xgb_model.pkl`) are on the [Hugging Face Space](https://huggingface.co/spaces/suhasadidela/Lung_Cancer_detection/tree/main).

---

## 📄 Publication

> *"A Stacking Ensemble-based Approach for Lung Cancer Detection"*  
> K. Suresh, H. Sai Karthik V, D. Vivek Reddy, **A. Suhas**, Pritam Khan  
> Department of Computational Intelligence, SRM Institute of Science and Technology

---

## 👥 Contributors

| Name | Role |
|---|---|
| **A. Suhas** ([@suhasadidela](https://github.com/suhasadidela)) | Data Collection & Preprocessing |
| H. Sai Karthik V ([@saikarthik0809](https://github.com/saikarthik0809)) | Model Development & Evaluation |
| D. Vivek Reddy | Literature Survey & Model Analysis |
| Dr. K. Suresh, Dr. Pritam Khan | Supervisors |

---

## 🔮 Future Work

- Attention mechanisms in CNN for better feature interpretability
- Domain adaptation and adversarial training
- Larger and more diverse datasets
- Real-time clinical deployment pipeline
- Explainable AI (XAI) integration
