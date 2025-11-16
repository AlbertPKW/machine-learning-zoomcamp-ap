# import the required packages
import numpy as np
import pandas as pd

# for modelling
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.encoding import OrdinalEncoder
from sklearn.pipeline import Pipeline
import pickle

def load_data():
    data_url = 'https://raw.githubusercontent.com/AlbertPKW/machine-learning-zoomcamp-ap/refs/heads/main/HR%20Analytics/hr_data_v2.csv'
    df = pd.read_csv(data_url) # Import dataset
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def train_model(df):
    dfy = df.is_promoted # model output
    dfX = df.drop(['is_promoted', 'employee_id'], axis=1) # Model Inputs

    # Split both Inputs (X) and Output (y) into training set (70%) and testing set (30%)
    X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.3, random_state=2)

    text_columns = ['department', 'education', 'gender', 'recruitment_channel']

    # Build Pipeline
    xgb_pipeline = Pipeline(steps=[
        ("imputer", ArbitraryNumberImputer(
            arbitrary_number=3,
            variables=['previous_year_rating']
        )),

        ("ordinal_enc", OrdinalEncoder(
            encoding_method='ordered',
            variables=text_columns
        )),

        ("model", XGBClassifier(
            n_estimators=200,
            min_child_weight=1,
            learning_rate=0.1,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=0.6,
            gamma=0.1,
            random_state=2,
            n_jobs=-1,
            tree_method='hist'
        ))
    ])

    # Fit the entire pipeline
    xgb_pipeline.fit(X_train, y_train)
    return xgb_pipeline

def save_model(filename, model):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f'Model saved to {filename}')

df = load_data()
pipeline = train_model(df)
save_model('xgb_model.bin', pipeline)