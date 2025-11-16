# Employee Promotion Prediction 

## Problem Statement

This project develops a machine-learning system to **predict whether an employee is likely to be promoted** during an internal HR evaluation cycle.  
The implementation includes data preprocessing, model training, and deployment as a FastAPI web service using the scripts:  
- Training pipeline: ```xgb_train.py```  
- API service: ```xgb_predict.py```  
- Prediction test script: ```xgb_test.py``` 

### What Problem Are We Solving?

HR teams often review thousands of employees each cycle. This makes promotion decisions:
- Time-consuming  
- Prone to inconsistency  
- Vulnerable to human bias  

This project automates part of the decision process by predicting the **probability that an employee will be promoted**, based on features such as:
- Demographics (age, gender, education)
- Performance (previous rating, KPI achievements, awards)
- Training history and scores  
- Job attributes (department, region, tenure)

The model outputs:
- `is_promoted_probability`
- `is_promoted` (threshold = 0.5)

The deployed FastAPI service (`/predict`) takes in employee data and returns these predictions in real time.

### Who Benefits From This Solution?

#### **HR Departments**
- Faster shortlisting of promotion candidates  
- More consistent and fair assessments  
- Reduced manual workload  

#### **Business Leaders & Managers**
- Better visibility of high-potential employees  
- Smarter succession planning  
- Data-driven resource allocation  

#### **Employees**
- Fairer and more transparent promotion processes  
- Increased trust in organizational decisions  

### How the Model Will Be Used

1. HR systems send employee info to the FastAPI endpoint.  
2. The model (loaded from `xgb_model.bin`) predicts:  
   - Probability of promotion  
   - Binary decision (promote / not promote)  
3. Results support HR in:  
   - Pre-screening  
   - Reviewing borderline promotion cases  
   - Enhancing decision fairness  

### Evaluation Metric — AUC-ROC

This problem is a **binary classification task** with **balanced** dataset.  
Therefore, the best evaluation metric is: **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**

**Why AUC-ROC?**
- Measures ranking ability between promoted vs. not promoted  
- Reflects the quality of predicted probabilities  
- Useful for threshold-based decision systems used in HR  

AUC-ROC provides a balanced and comprehensive view of model performance across all thresholds.

### Why This Problem Matters

Promotion decisions impact:
- Employee morale  
- Retention and turnover  
- Career development  
- Organizational trust and fairness  

A data-driven solution:
- Reduces unconscious bias  
- Improves transparency  
- Ensures consistent standards  
- Helps HR invest in the right people  

By supporting fair and efficient promotion decisions, the model directly contributes to a healthier and more productive workplace.

---

## Data Preparation & EDA

### 1. Data Source and Retrieval

The dataset used in this project is an HR employee dataset containing demographics, performance history, training scores, and promotion outcomes.

Data loading and initial preprocessing steps are fully reproducible, as implemented in the `load_data()` function inside **xgb_train.py**:

```python
def load_data():
    data_url = 'https://raw.githubusercontent.com/AlbertPKW/machine-learning-zoomcamp-ap/refs/heads/main/HR%20Analytics/hr_data_v2.csv'
    df = pd.read_csv(data_url)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df
```

#### Key Points
- Dataset retrieved from a public GitHub URL.
- Column names standardized to lowercase snake_case.
- Ensures consistency and reproducibility in training.

### 2. Exploratory Data Insights

Summary of data exploration before model training.

#### 2.1 Numerical Features — Distribution & Outliers

(Boxplots and histograms were used)

##### **no_of_trainings**
- Right-skewed with most employees having 1 training.
- Outliers at 5–8 trainings.

##### **age**
- Mostly ages 28–36.
- Outliers above 50.

##### **previous_year_rating**
- Clusters at 3, 4, 5.
- Some missing values handled during preprocessing.

##### **length_of_service**
- Strong right-skew.
- Most under 10 years; outliers up to 35 years.

##### **avg_training_score**
- Multimodal distribution between 40 and 100.

#### 2.2 Categorical Features — Proportions & Imbalance

From bar charts:

- **Department**: Largest groups are Sales & Marketing, Operations, Technology.
- **Region**: Highly unbalanced; Region 2 and 22 dominate.
- **Education**: Mostly Bachelor’s.
- **Gender**: ~65% Male.
- **Recruitment Channel**: Mostly Other and Sourcing.
- **KPIs_met_>80%**: Roughly 50/50.
- **Awards_won?**: Highly imbalanced (over 90% have no awards).

#### 2.3 Missing Values

- Most columns have no missing values.
- **previous_year_rating** contains missing entries.

Handled via:

```python
ArbitraryNumberImputer(arbitrary_number=3, variables=['previous_year_rating'])
```
Ensures reproducible imputation.

#### 2.4 Correlations Between Numerical Variables

From heatmap:

- **age ↔ length_of_service** moderately correlated (~0.66).
- All other correlations very weak (<0.1).
- No multicollinearity issues.

#### 2.5 Distribution Shapes (Histograms)

- **no_of_trainings** and **length_of_service** extremely skewed.
- **previous_year_rating** is discrete.
- **avg_training_score** shows multiple peaks.

These characteristics validate the choice of XGBoost (robust to skewness and outliers).

### 3. Reproducible Preprocessing Across Scripts

#### **xgb_train.py**  
Contains the end-to-end preprocessing pipeline:

```python
Pipeline(steps=[
    ("imputer", ArbitraryNumberImputer(...)),
    ("ordinal_enc", OrdinalEncoder(...)),
    ("model", XGBClassifier(...))
])
```

#### **xgb_predict.py**  
Loads the **same serialized pipeline** (`xgb_model.bin`) ensuring identical preprocessing at inference time.

#### **xgb_test.py**  
Sends inference requests to the API to validate preprocessing consistency.

The combined workflow ensures the exact same preprocessing logic is applied in:
- Data Preprocessing
- Model training
- API inference
- External testing

---

## Modeling Approach & Evaluation Metrics

### 1. Baseline Model

The modeling process begins with a simple **baseline classifier**, which is a default **XGBoost** model **without tuning**

The goal of the baseline is to establish a reference point for later comparison.  
It helps determine whether more sophisticated algorithms and tuning truly add predictive value.


### 2. Train Multiple Models

At least **three** different models were trained in the notebook before selecting the final candidate. These include:

#### **1st Model: Baseline Model**
- Default parameters  
- Minimal preprocessing  
- Provides foundational metrics for comparison  

#### **2nd Model: Fine-Tuned Model**
Using insights from the data distribution and early model performance:
- Adjusted learning rate  
- Modified tree depth  
- Tuned number of estimators  
- Set gamma, subsample, and column sampling ratios  
- Performed targeted hyperparameter tuning via manual grid iterations  

This model typically showed strong improvement in recall and F1 score while maintaining stable accuracy.

#### **3rd Model: Feature-Importance-Driven Model**
Based on feature importance results:
- Removed weak or noisy features  
- Focused on top-impact predictors (e.g., previous_year_rating, KPIs_met_>80%, avg_training_score)  
- Retrained model to evaluate whether dimensionality reduction improved generalization  

#### **4th Model: Scaling / Normalization Variant**
Although tree-based models are scale-invariant, a scaled variant was evaluated for completeness:
- Applied MinMaxScaler
- Retrained model to compare with unscaled version  

This confirmed that scaling had minimal impact, as expected for XGBoost.

#### **5th Model: K-Fold Cross-Validation Model**
To improve robustness:
- Performed **K-Fold Cross-Validation (K = 5)**  
- Ensured the model generalizes across multiple splits  
- Averaged metrics used to confirm consistency  

### 3. Cross-Validation

Cross-validation was used to:
- Reduce model variance  
- Ensure performance generalizes to unseen data  
- Compare models fairly  

Typical metrics computed during CV:
- Accuracy  
- Precision  
- Recall  
- F1 score  
- ROC-AUC  

#### 4. Summary of Model Performance

Below is the final summary table of all evaluated models:

| Model               | Testing Accuracy | Precision | Recall | F1 score | ROC-AUC |
| ------------------- | ---------------- | --------- | ------ | -------- | ------- |
| Base Model          | 0.811            | 0.786     | 0.866  | 0.824    | 0.899   |
| Fine-Tuned          | 0.825            | 0.780     | 0.915  | 0.842    | 0.905   |
| Feature Importance  | 0.816            | 0.767     | 0.918  | 0.836    | 0.902   |
| Scaling             | 0.816            | 0.767     | 0.918  | 0.836    | 0.902   |
| K-Fold              | 0.905            | -         | -      | -        | -       |

#### 5. Model Selection Strategy

The final model was chosen using the following criteria:
- **Highest ROC-AUC** 
- **Strong F1 Score** 

The **Fine-Tuned XGBoost Model** showed the strongest overall performance in terms of F1 score and ROC-AUC score.  

---

## How the System Works — Training, Serving, and Testing

This project is organized into three core scripts that work together to train the machine-learning model, deploy it via an API, and verify that predictions work correctly.  
Each script has a clearly defined responsibility, ensuring the entire workflow is **modular**, **reproducible**, and **easy to maintain**.

### 1. `xgb_train.py` — Model Training Pipeline

This script is responsible for **end-to-end model development**. It contains all components needed to train a fully reproducible XGBoost classification model.

#### **Key Steps Performed**
##### **1. Load the dataset**
The `load_data()` function retrieves the HR dataset from GitHub and standardizes column names.

##### **2. Preprocess the input features**
A scikit-learn `Pipeline` is built with:
- `ArbitraryNumberImputer` — handles missing values in `previous_year_rating`
- `OrdinalEncoder` — converts categorical variables into numeric ordered labels
- `XGBClassifier` — the final machine-learning model

This ensures the **same preprocessing rules** are applied during both training and inference.

##### **3. Train the final model**
The pipeline is trained using a 70/30 train-test split.  
Hyperparameters are configured to match the fine-tuned model from the notebook.

##### **4. Save the trained model**
Once training is complete, the entire pipeline (not just the model) is serialized:

```python
with open('xgb_model.bin', 'wb') as f_out:
    pickle.dump(model, f_out)
```

This ensures:
- Preprocessing + model are stored together  
- Inference script (`xgb_predict.py`) reloads **the exact same pipeline**

### 2. `xgb_predict.py` — FastAPI Web Service for Inference

This script converts the trained model into a production-ready **REST API** that can receive JSON input and return promotion predictions.

#### **Key Steps Performed**
##### **1. Load the saved model**
The script loads the serialized pipeline from `xgb_model.bin`:

```python
with open('xgb_model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
```

Since the pipeline includes imputers, encoders, and the classifier, no manual preprocessing is required.

##### **2. Define a request schema using Pydantic**
The `Employee` model:
- Ensures correct data types  
- Renames fields like `kpis_met_>80%` using aliasing  
- Forbids extra/unexpected fields  

This guarantees input validation and prevents malformed requests.

##### **3. Create the prediction endpoint**
The `/predict` POST endpoint:
- Converts JSON input into a pandas DataFrame  
- Passes it through the pipeline’s `.predict_proba()` method  
- Returns:
  - probability of promotion  
  - binary decision (>= 0.5 threshold)

Example output:
```json
{
  "is_promoted_probability": 0.7421,
  "is_promoted": true
}
```

##### **4. Start the FastAPI service**
The script can be run locally with:

```
uvicorn xgb_predict:app --host 0.0.0.0 --port=9797
```

This exposes the ML model as a web service ready for consumption.

### 3. `xgb_test.py` — External Test Script for Verification

This script acts as a **client** that sends sample employee data to the running API to confirm that predictions are functioning correctly.

#### **Key Steps Performed**
##### **1. Define a sample employee JSON payload**
This mirrors the expected Pydantic schema used by the API.

##### **2. Send a request to the running FastAPI service**
Using Python’s `requests` library:
```python
response = requests.post(url, json=employee)
```

The script supports testing:
- Local deployment  
- Cloud deployment (e.g., Fly.io)

##### **3. Display the model’s response**
It prints:
- The raw API response  
- A human-friendly message (e.g., "Employee is likely to be promoted")

This ensures:
- The API is reachable  
- Input formatting is correct  
- The model and preprocessing pipeline load correctly  
- End-to-end prediction works as intended

`xgb_test.py` does *not* start the API itself — it assumes `xgb_predict.py` is already running.  
It is purely an **integration test tool**.


### End-to-End Workflow Summary

**1. Train → 2. Serve → 3. Test**

| Step | Script | Description |
|------|--------|-------------|
| **Model Training** | `xgb_train.py` | Builds the pipeline, trains XGBoost, saves `xgb_model.bin` |
| **Model Serving** | `xgb_predict.py` | Loads the bin file and exposes `/predict` API endpoint |
| **Model Testing** | `xgb_test.py` | Sends sample requests to API to verify predictions |
