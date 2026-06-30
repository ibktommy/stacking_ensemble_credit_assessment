# Developing a Heterogeneous Stacking Ensemble Machine Learning Framework for Multi-Class Credit Risk Assessment

### An advanced machine learning web application built to predict borrower default risk tiers accurately.
**Live Web Application Link:** [https://adewale-credit-risk-assessment-app.streamlit.app/](https://adewale-credit-risk-assessment-app.streamlit.app/)

---

## 📌 Project Overview

I developed an end-to-end machine learning system that transforms how banks evaluate loan risk. By combining two distinct credit datasets, cleaning data anomalies, and applying smart oversampling, I built a reliable foundation. I then trained multiple models to classify applicants into four unique risk priority tiers.

My final architecture uses a powerful stacking ensemble technique. Three diverse baseline models pass their initial predictions to a final XGBoost coordinator, achieving incredible classification performance. I successfully deployed this entire pipeline into an interactive, live web dashboard for instant financial risk decision-making.

---

## 🛠️ Research Methodology & Core Pipeline

To build this project, I structured the workflow into distinct, sequential engineering phases:

### 1. Data Integration and Preprocessing
* **Source Merging:** I combined two isolated institutional credit datasets using their unique identifier keys into a single master matrix of 51,336 historical applications.
* **Cleaning & Filtering:** I identified missing data and completely eliminated columns suffering from extreme multicollinearity (high multi-variable correlation) to ensure raw feature quality.
* **Feature Scaling:** I applied a mathematical `StandardScaler` to bring variance-heavy continuous metrics (like monthly income and credit scores) into a uniform range, preventing data skew.

### 2. Resolving Class Imbalance
The banking dataset was severely lopsided, as standard borrowers heavily outnumbered high-risk defaults. To prevent my model from defaulting blindly to the majority class, I implemented **SMOTE (Synthetic Minority Over-sampling Technique)**. This technique artificially balanced the training matrix up to an even 90,156 rows across all four target risk tiers.

### 3. Dual-Engine Feature Selection
To optimize the system's speed and remove data noise, I filtered the original 72 columns down to the **Top 35 most predictive features** by blending two separate statistical strategies:
* **Mutual Information Scoring:** Measuring non-linear relationships and shared entropy between individual variables and the target risk tier.
* **Random Forest Gini Importance:** Evaluating total purity gains across tree splits.

---

## 🤖 Model Architecture & Stacking Ensemble

Instead of relying on a single algorithm, my project relies on a **Heterogeneous Stacking Framework**. This layered architecture uses a multi-model setup to catch different data patterns:

[Applicant Input Data]
│
├──► Model 1: Logistic Regression (Captures Linear Boundaries)
├──► Model 2: Random Forest (Handles Parallel Data Subsets)
└──► Model 3: LightGBM (Learns Sequentially from Errors)
│
▼  [Out-of-Fold Probability Vectors]
[Level-1 Meta-Learner: XGBoost]
│
▼
[Final Multi-Class Risk Output (P1, P2, P3, or P4)]

### Preventing Data Leakage
To train the system safely without "data cheating", I used **5-Fold Cross-Validation Out-of-Fold (OOF)** prediction matrices. This ensures that when the final meta-learner learns how to combine the base models' outputs, it only uses predictions made on data the models have never seen before.

---

## 📈 Performance & Empirical Evaluation

I ran a rigorous performance audit on a 30% untouched holdout test set to compare my Stacked Ensemble against the individual standalone base learners.

### Master Performance Comparison Table
| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7841 | 0.7214 | 0.7350 | 0.7281 |
| Random Forest | 0.8129 | 0.7640 | 0.7511 | 0.7575 |
| LightGBM | 0.8410 | 0.7983 | 0.8104 | 0.8043 |
| **Stacked Ensemble (Final)**| **0.8594** | **0.8211** | **0.8329** | **0.8269** |

### Per-Class F1-Score Breakdown
| Model Architecture | P1 (Low Risk) | P2 (Med-Low) | P3 (Med-High) | P4 (High Risk) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7420 | 0.8115 | 0.6902 | 0.6687 |
| Random Forest | 0.7719 | 0.8354 | 0.7210 | 0.7017 |
| LightGBM | 0.8145 | 0.8601 | 0.7812 | 0.7615 |
| **Stacked Ensemble (Final)**| **0.8388** | **0.8792** | **0.8014** | **0.7882** |

### Key Findings
* **Ensemble Superiority:** The Stacked Ensemble achieved the highest overall metrics, reaching a **Macro F1-Score of 0.8269** and **Accuracy of 85.94%**.
* **Minority Class Accuracy:** Thanks to SMOTE, the F1-scores for difficult high-risk classes (P3 and P4) improved significantly, proving the model does not just favor the majority class.

---

## 🚀 Web Deployment & Interactive Interface

To bring this research to life, I built an interactive web dashboard using **Streamlit** and hosted it on the cloud. 

### Core Features of the Web App:
* **Manual Stress Testing:** Users can adjust custom input sliders for an applicant's credit score, net income, age, and loan history to watch the model update its risk assessment in real time.
* **Authentic Test Mode:** The app includes a dropdown menu connected directly to an independent pool of 100 real-world holdout test records. Selecting a row automatically populates the fields and displays the true credit outcome next to the model's prediction chart.

---

## 💻 How to Run the App Locally

If you want to run this web application on your local machine, follow these simple steps:

1. Clone this repository to your computer:
   git clone [https://github.com/your-username/credit-risk-stacking-app.git](https://github.com/your-username/credit-risk-stacking-app.git)
   cd credit-risk-stacking-app

2. Install all required dependencies using the project text file:
   pip install -r requirements.txt

3. Boot up the local Streamlit application server:
   streamlit run app.py