# 🏠 Real Estate Valuation & Investment Analyzer

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Machine Learning system** that transforms real estate data into actionable investment insights, enabling smarter and data-driven property decisions.

---

## 🚀 Project Overview

This project combines **predictive modeling + investment analysis** to evaluate real estate properties.
It not only predicts property prices but also provides **ROI estimation, risk scoring, and portfolio optimization**.

---

## 📸 Screenshots

<p align="center">
  <img src="screenshots/dashboard.png" width="800"/>
</p>

<p align="center">
  <img src="screenshots/prediction.png" width="800"/>
</p>

<p align="center">
  <img src="screenshots/analysis.png" width="800"/>
</p>

---

## 📊 Key Results

| Metric                 | Value  |
| ---------------------- | ------ |
| Valuation Accuracy     | 99.7%  |
| Prediction Error (MAE) | $910   |
| R² Score               | 0.9972 |
| Properties Analyzed    | 1,460  |
| Buy Recommendations    | 43     |

---

## ✨ Features

### 📊 Real Estate Valuation

* Predicts property prices with high accuracy
* Identifies key value-driving factors
* Handles multiple property attributes

### 💰 Investment Analysis

* Investment Score (0–100)
* ROI potential calculation
* Risk assessment
* Rental yield estimation

### 📈 Portfolio Optimization

* Selects best properties within budget
* Balances risk vs return
* Maximizes investment score

### 🎯 Recommendations

* BUY: High potential properties
* CONSIDER: Moderate opportunities
* HOLD: Low priority investments

---

## 🏗️ Project Architecture

```
real-estate-valuation-investment-analyzer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   ├── 02_Model_Building.ipynb
│   └── 03_Investment_Analysis_Engine.ipynb
│
├── app/
│   └── app.py
│
├── models/
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Models

* Random Forest
* XGBoost
* LightGBM
* Ensemble (Stacking)

---

## 📈 Model Performance

| Model         | MAE   | R² Score |
| ------------- | ----- | -------- |
| Ensemble      | $910  | 0.9972   |
| Random Forest | $958  | 0.9906   |
| XGBoost       | $3348 | 0.9899   |
| LightGBM      | $5662 | 0.9583   |

---

## 🎯 Key Insights

* Basement size and construction year strongly influence price
* Newer properties have higher valuation
* Location significantly impacts investment potential

---

## 🛠️ Tech Stack

**Machine Learning:**
Python, Scikit-learn, XGBoost, LightGBM

**Data Processing:**
Pandas, NumPy

**Visualization:**
Matplotlib, Seaborn, Plotly

**Dashboard:**
Streamlit

---

## 🚀 How to Run

```bash
git clone https://github.com/priya-tiwarii/real-estate-valuation-investment-analyzer.git
cd real-estate-valuation-investment-analyzer

conda create -n real_estate python=3.9 -y
conda activate real_estate

pip install -r requirements.txt

streamlit run app/app.py
```

---

## 📌 Dataset

House Prices Dataset (Kaggle)
Includes features such as size, location, year built, and more.

---

## 🔮 Future Improvements

* Deploy on Streamlit Cloud
* Add real-time property data
* Improve UI/UX
* Integrate advanced models

---

## 👩‍💻 Author

**Priya Tiwari**
GitHub: https://github.com/priya-tiwarii

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐
