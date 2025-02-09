## 📊 Advanced Churn Prediction with Deep Learning & Active Learning

## 🚀 Overview

This project implements an AI-driven customer churn prediction system that uses a hybrid Deep Learning + XGBoost approach. It integrates active learning, concept drift detection, and Ray Tune for hyperparameter optimization to enhance performance over time.

## 🛠️ Features

✅ Synthetic Data Generation – Generates realistic telecom customer data.
✅ Advanced Preprocessing – Feature engineering, SMOTE balancing, and robust scaling.
✅ Hybrid AI Model – Combines Deep Learning with XGBoost for accuracy.
✅ Hyperparameter Tuning – Uses Ray Tune for optimized batch sizes and learning rates.
✅ Active Learning – Dynamically retrains on uncertain data points.
✅ Concept Drift Detection – Detects data distribution shifts using Wasserstein Distance.
✅ Visualization – Displays prediction distribution and ROC curves.

## 📂 Project Structure
```bash
📁 advanced-churn-prediction
│── 📄 README.md                # Documentation  
│── 📄 requirements.txt         # Dependencies  
│── 📄 .gitignore               # Files to ignore in Git  
│── 📂 data/                    # Dataset placeholder  
│── 📂 models/                  # Saved trained models  
│── 📄 preprocess.py            # Data processing & feature engineering  
│── 📄 train.py                 # Model training & tuning  
│── 📄 active_learning.py       # Real-time learning & model updates  
│── 📄 drift_detection.py       # Concept drift monitoring  
│── 📄 visualize.py             # Model evaluation & plotting
```

## 📦 Installation
🔹 Clone the Repository
```bash
git clone https://github.com/yourusername/advanced-churn-prediction.git
cd advanced-churn-prediction
```

🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

🔹 Set Up Ray Tune (for Hyperparameter Optimization)
```bash
ray start --head
```

## 📊 Data Preprocessing

Run the preprocessing script to generate synthetic customer data and apply feature engineering:

```bash

python preprocess.py
```

## 🤖 Model Training

Train the deep learning model with optimized hyperparameters:

```bash
python train.py
```

This script will:
✅ Split Data into training and validation sets
✅ Perform Hyperparameter Tuning using Ray Tune
✅ Train the Best Model based on performance
✅ Save the Model in the models/ directory

## 🧠 Active Learning for Model Improvement

The model dynamically improves by retraining on uncertain predictions:

```bash
python active_learning.py
```

## 📡 Concept Drift Detection

Detect if the input data distribution has changed significantly:

```bash
python drift_detection.py
```

## 📈 Model Evaluation & Visualization

Generate performance metrics and visualizations:

```bash
python visualize.py
```

This script produces:
📊 Prediction Distribution
📉 ROC Curve

## 📜 Example Output

📊 Churn Prediction Distribution

# ##📉 ROC Curve

📌 Future Enhancements
🚀 Integrate AutoML for feature selection
🚀 Deploy as a REST API for real-time inference
🚀 Add Explainability (SHAP) for feature importance analysis


### 📮 Contact

For questions or suggestions, reach out at:
📧 Email: raghavmrparadise@gmail.com
