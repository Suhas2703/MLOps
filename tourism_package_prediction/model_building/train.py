# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
from huggingface_hub import hf_hub_download

mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_tracking_uri("file:///content/mlruns")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

repo_id_data = "SuhasKashyap2703/Tourism-Package-Prediction"
repo_type_data = "dataset"

# Xtrain_path = "hf://datasets/SuhasKashyap2703/Tourism-Package-Prediction/Xtrain.csv"
# Xtest_path = "hf://datasets/SuhasKashyap2703/Tourism-Package-Prediction/Xtest.csv"
# ytrain_path = "hf://datasets/SuhasKashyap2703/Tourism-Package-Prediction/ytrain.csv"
# ytest_path = "hf://datasets/SuhasKashyap2703/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(
    hf_hub_download(repo_id=repo_id_data, filename="Xtrain.csv", repo_type=repo_type_data)
)

Xtest = pd.read_csv(
    hf_hub_download(repo_id=repo_id_data, filename="Xtest.csv", repo_type=repo_type_data)
)

ytrain = pd.read_csv(
    hf_hub_download(repo_id=repo_id_data, filename="ytrain.csv", repo_type=repo_type_data)
)

ytest = pd.read_csv(
    hf_hub_download(repo_id=repo_id_data, filename="ytest.csv", repo_type=repo_type_data)
)

# Xtrain = pd.read_csv(Xtrain_path)
# Xtest = pd.read_csv(Xtest_path)
# ytrain = pd.read_csv(ytrain_path)
# ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',               # Age of the customer
    'DurationOfPitch',   # Duration of the sales pitch delivered to the customer.
    'NumberOfTrips',     # Average number of trips the customer takes annually.
    'MonthlyIncome',     # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # The method by which the customer was contacted
    'CityTier',              # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'Occupation',            # Customer's occupation
    'Gender',                # Gender of the customer
    'NumberOfPersonVisiting', # Total number of people accompanying the customer
    'NumberOfFollowups',     # Total number of follow-ups by the salesperson
    'ProductPitched',        # The type of product pitched to the customer
    'PreferredPropertyStar', # Preferred hotel rating by the customer
    'MaritalStatus',         # Marital status of the customer
    'Passport',              # Whether the customer holds a valid passport or not
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch
    'OwnCar',                # Whether the customer owns a car or not
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer.
    'Designation'            # Customer's designation in their current organization
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="logloss",
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbclassifier__n_estimators": [100, 150],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.5],
}

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )

    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    threshold = 0.45

    ytrain_pred = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    ytest_pred = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, ytrain_pred, output_dict=True)
    test_report = classification_report(ytest, ytest_pred, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    model_path = "best_churn_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

api = HfApi()
model_repo_id = "SuhasKashyap2703/churn-model"

try:
    api.repo_info(repo_id=model_repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=model_repo_id, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=model_repo_id,
    repo_type="model",
)

print("Training completed successfully.")
print("Model uploaded to Hugging Face.")
