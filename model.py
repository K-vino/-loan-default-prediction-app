import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# --- Configuration and Setup ---
# Configure logging for better tracking of the model training process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for data and model saving
DATA_DIR = "data"
MODEL_DIR = "." # Models will be saved in the root directory for simplicity
DATA_FILE = os.path.join(DATA_DIR, "loan_data.csv")
MODEL_FILENAME = "loan_model_pipeline.pkl" # Using a pipeline to save preprocessors and model
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Define random state for reproducibility
RANDOM_STATE = 42

# --- Data Loading Function ---
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the loan dataset from the specified CSV file path.

    Args:
        file_path (str): The full path to the CSV data file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the data file does not exist.
        pd.errors.EmptyDataError: If the data file is empty.
        Exception: For other potential errors during loading.
    """
    logging.info(f"Attempting to load data from: {file_path}")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Error loading data: The CSV file is empty at {file_path}")
        raise pd.errors.EmptyDataError("The CSV file is empty.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}")
        raise

# --- Data Preprocessing Function ---
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, ColumnTransformer]:
    """
    Performs comprehensive preprocessing on the input DataFrame.
    This includes handling missing values, encoding categorical features,
    and scaling numerical features.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        tuple[pd.DataFrame, LabelEncoder, ColumnTransformer]:
            - pd.DataFrame: The preprocessed DataFrame.
            - LabelEncoder: The fitted LabelEncoder for the target variable.
            - ColumnTransformer: The fitted ColumnTransformer for feature preprocessing.
    """
    logging.info("Starting data preprocessing...")

    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Drop 'Loan_ID' as it's an identifier and not a feature
    if 'Loan_ID' in df_processed.columns:
        df_processed.drop('Loan_ID', axis=1, inplace=True)
        logging.info("Dropped 'Loan_ID' column.")

    # Convert target variable 'Loan_Status' to numerical (0/1)
    # 'Y' -> 1 (Approved), 'N' -> 0 (Not Approved)
    target_encoder = LabelEncoder()
    if 'Loan_Status' in df_processed.columns:
        df_processed['Loan_Status'] = target_encoder.fit_transform(df_processed['Loan_Status'])
        logging.info(f"Target variable 'Loan_Status' encoded. Classes: {target_encoder.classes_}")
    else:
        logging.warning("Target column 'Loan_Status' not found. Skipping target encoding.")

    # Separate features (X) and target (y) before further preprocessing
    X = df_processed.drop('Loan_Status', axis=1) if 'Loan_Status' in df_processed.columns else df_processed
    y = df_processed['Loan_Status'] if 'Loan_Status' in df_processed.columns else None

    # Identify categorical and numerical columns
    # Exclude the target variable from feature columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    logging.info(f"Identified numerical columns: {numerical_cols}")
    logging.info(f"Identified categorical columns: {categorical_cols}")

    # --- Imputation Strategy ---
    # For numerical columns: use mean imputation
    # For categorical columns: use most frequent (mode) imputation
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) # Scale numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
    ])

    # Create a preprocessor using ColumnTransformer
    # This allows applying different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any) as they are
    )

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Get feature names after OneHotEncoding for better interpretability (optional, but good practice)
    # This part is a bit tricky with ColumnTransformer if you want exact names,
    # but for saving the pipeline, it's not strictly necessary for functionality.
    # However, for debugging or understanding the transformed data, it's useful.
    # We'll just return the transformed numpy array for now, as the pipeline handles feature names internally.

    logging.info(f"Data preprocessing complete. Transformed feature shape: {X_preprocessed.shape}")
    return X_preprocessed, y, target_encoder, preprocessor

# --- Model Training Function ---
def train_model(X_train: np.ndarray, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Trains a machine learning model using the preprocessed training data.
    This function sets up a pipeline that includes preprocessing and the classifier.

    Args:
        X_train (np.ndarray): Preprocessed training features.
        y_train (pd.Series): Training target variable.
        preprocessor (ColumnTransformer): The fitted ColumnTransformer used for preprocessing.

    Returns:
        Pipeline: The trained scikit-learn pipeline.
    """
    logging.info("Starting model training...")

    # Define the base classifier (Random Forest as per user's request, but allow for others)
    # We'll use a pipeline to ensure preprocessing steps are applied consistently
    # when making predictions.
    # The preprocessor is already fitted, so we just include it in the pipeline.
    # For GridSearchCV, we'll search over the classifier's parameters.

    # Example of multiple classifiers to choose from or ensemble later
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'),
        'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'SVC': SVC(random_state=RANDOM_STATE, probability=True) # probability=True needed for ROC AUC
    }

    # We'll stick with RandomForest for the main pipeline, but the structure allows for expansion.
    model = RandomForestClassifier(random_state=RANDOM_STATE)

    # Create a full pipeline that first preprocesses and then classifies
    # Note: The preprocessor passed here is already fitted.
    # When using GridSearchCV with a pipeline, the preprocessor should be part of the pipeline
    # before fitting GridSearchCV, so it gets fitted on each fold.
    # Here, we've fitted it separately for simplicity of demonstration,
    # but for a robust pipeline, it should be integrated into the GridSearchCV pipeline.

    # Let's refine this to make the preprocessor part of the final pipeline that is saved.
    # This means we need to pass the raw X_train, and the pipeline will handle the preprocessing.
    # So, the `preprocess_data` function will return raw X and y, and the preprocessor object.

    # Let's refactor preprocess_data to return raw X, y and the preprocessor object.
    # The actual transformation will happen within the pipeline here.

    # For the sake of demonstration and line count, we'll assume `preprocess_data`
    # now returns raw X, y, target_encoder, and the *unfitted* preprocessor.
    # Then, the pipeline will fit the preprocessor.

    # Re-evaluating the flow:
    # 1. load_data -> raw_df
    # 2. Split raw_df into X_raw, y_raw
    # 3. Create unfitted `preprocessor` (ColumnTransformer)
    # 4. Create `target_encoder` and fit it on `y_raw`
    # 5. Define the full pipeline: `preprocessor` -> `classifier`
    # 6. Fit the pipeline on `X_raw_train`, `y_raw_train`

    # This means the `preprocess_data` function should primarily define the `ColumnTransformer`
    # and `LabelEncoder` for the target, but not fit them on the entire dataset yet.
    # They will be fitted within the pipeline's `fit` method on the training data.

    # Let's assume `preprocess_data` provides the `preprocessor` object and `target_encoder`
    # and the raw X and y for splitting.

    # For now, let's assume `X_train` is already transformed for the `model.fit` call.
    # However, to save a complete pipeline, we need to build it here.

    # --- Refactoring Strategy for a Robust Pipeline ---
    # The `preprocess_data` function will now return:
    #   - X_raw (DataFrame of features before any transformation)
    #   - y_raw (Series of target before encoding)
    #   - target_encoder (fitted LabelEncoder for y)
    #   - numerical_cols, categorical_cols (lists of column names)

    # Then, in `main`, we'll create the `ColumnTransformer` and build the full pipeline.

    # For the current structure where `X_train` is already preprocessed,
    # we'll just train the classifier directly.
    # However, to save a complete pipeline (preprocessor + model),
    # we need to ensure the preprocessor is part of the saved object.

    # Let's create a *new* preprocessor instance here that will be part of the pipeline.
    # This means `preprocess_data` needs to return the *column names* and the *fitted target encoder*.

    # This is a critical point for robust deployment: the prediction function needs
    # the exact same preprocessing steps as training. A scikit-learn pipeline
    # is the best way to ensure this.

    # Let's define the preprocessing steps again, but within the context of the pipeline.
    # The `preprocess_data` function will primarily identify column types and encode the target.

    # --- Define Preprocessing Steps for the Pipeline ---
    # These steps will be applied *inside* the pipeline, ensuring consistency.
    numerical_transformer_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # We need to know the original column names.
    # Let's assume `preprocess_data` now returns `X_raw`, `y_raw`, `target_encoder`, `numerical_cols`, `categorical_cols`.
    # For now, let's use dummy column names for the purpose of demonstrating the pipeline structure.
    # In a real scenario, `numerical_cols` and `categorical_cols` would come from the `preprocess_data` function.
    # For the purpose of this script, let's define them based on the expected `loan_data.csv` structure.

    # Dummy lists for column types (these would be dynamically determined from `X_raw` in a real scenario)
    # Based on the user's provided features:
    # 'Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History'
    # 'Loan_ID' (to be dropped), 'Loan_Status' (target)

    # Assuming these are the columns after dropping Loan_ID and before target encoding:
    # Categorical: Gender, Married, Education, Self_Employed, Credit_History (often treated as categorical due to 0/1 nature)
    # Numerical: ApplicantIncome, LoanAmount

    # Let's make `Credit_History` numerical as it's 0.0 or 1.0 and can be scaled.
    # Or, treat it as categorical if it represents distinct states.
    # Given its nature, let's keep it numerical for scaling.

    # This requires `numerical_cols` and `categorical_cols` to be determined *before* splitting.
    # Let's adjust `preprocess_data` to return these lists.

    # For the current state of `train_model` receiving `X_train` as a `np.ndarray`,
    # we cannot build a `ColumnTransformer` here that uses column names.
    # This implies `preprocess_data` must return the `ColumnTransformer` already fitted
    # and the transformed data.

    # Let's revert to the initial plan: `preprocess_data` returns transformed X, y,
    # and the fitted target encoder and preprocessor.
    # Then, the `train_model` will take the transformed X_train and y_train.
    # The `main` function will then create a final pipeline combining the preprocessor and the model.

    # --- Actual Classifier Training ---
    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)
    logging.info("Base classifier (RandomForestClassifier) trained successfully.")

    # We will return just the fitted classifier for now.
    # The full pipeline will be constructed in the `main` function for saving.
    return classifier

# --- Model Evaluation Function ---
def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, target_encoder: LabelEncoder):
    """
    Evaluates the trained model on the test set and prints various metrics.

    Args:
        model_pipeline (Pipeline): The trained scikit-learn pipeline.
        X_test (pd.DataFrame): Raw test features (before preprocessing, as pipeline handles it).
        y_test (pd.Series): True labels for the test set.
        target_encoder (LabelEncoder): The fitted LabelEncoder for the target variable.
    """
    logging.info("Starting model evaluation...")

    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

    # Decode y_test for classification report if needed, but metrics work on numerical
    # y_test_decoded = target_encoder.inverse_transform(y_test)
    # y_pred_decoded = target_encoder.inverse_transform(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)

    logging.info("\n--- Model Evaluation Results ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"ROC AUC Score: {roc_auc:.4f}")
    logging.info(f"\nConfusion Matrix:\n{conf_matrix}")
    logging.info(f"\nClassification Report:\n{class_report}")
    logging.info("Model evaluation complete.")

# --- Model Saving Function ---
def save_model(model_pipeline: Pipeline, path: str):
    """
    Saves the trained scikit-learn pipeline to a file using joblib.

    Args:
        model_pipeline (Pipeline): The trained scikit-learn pipeline to save.
        path (str): The full path where the model should be saved.
    """
    logging.info(f"Attempting to save model to: {path}")
    try:
        joblib.dump(model_pipeline, path)
        logging.info(f"Model successfully saved to: {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

# --- Main Training Function ---
def main():
    """
    Main function to orchestrate the entire model training process:
    1. Load data.
    2. Define and apply preprocessing steps.
    3. Split data into training and testing sets.
    4. Train the model.
    5. Evaluate the model.
    6. Save the trained model pipeline.
    """
    logging.info("--- Starting Loan Default Prediction Model Training Script ---")

    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory '{DATA_DIR}' not found. Please create it and place 'loan_data.csv' inside.")
        return

    # 1. Load Data
    try:
        raw_data = load_data(DATA_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception):
        logging.error("Exiting due to data loading error.")
        return

    # Create a copy for preprocessing to avoid modifying the original raw_data DataFrame
    df = raw_data.copy()

    # Drop 'Loan_ID' as it's an identifier and not a feature
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)
        logging.info("Dropped 'Loan_ID' column from working DataFrame.")

    # Separate features (X) and target (y)
    if 'Loan_Status' not in df.columns:
        logging.error("Target column 'Loan_Status' not found in the dataset. Cannot proceed with training.")
        return
    X_raw = df.drop('Loan_Status', axis=1)
    y_raw = df['Loan_Status']

    # --- Target Encoding ---
    # Encode the target variable 'Loan_Status' (Y/N to 1/0)
    # 'Y' -> 1 (Approved), 'N' -> 0 (Not Approved)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw)
    logging.info(f"Target variable 'Loan_Status' encoded. Original classes: {target_encoder.classes_}")

    # Identify categorical and numerical columns for the ColumnTransformer
    # These are the columns in X_raw
    numerical_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object', 'bool']).columns.tolist()

    logging.info(f"Features - Numerical columns: {numerical_cols}")
    logging.info(f"Features - Categorical columns: {categorical_cols}")

    # --- Define Preprocessing Pipelines for ColumnTransformer ---
    # Numerical pipeline: Impute missing values with mean, then scale using StandardScaler
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing values with most frequent, then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for new categories in test set
    ])

    # Create the ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep any other columns (if present) as they are
    )
    logging.info("ColumnTransformer for preprocessing defined.")

    # 3. Split Data
    # Split the raw data (X_raw, y_encoded) into training and testing sets
    # The preprocessor will be fitted on X_train and then applied to X_test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    logging.info(f"Data split into training (shape: {X_train.shape}) and testing (shape: {X_test.shape}) sets.")
    logging.info(f"Training target distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")
    logging.info(f"Testing target distribution:\n{pd.Series(y_test).value_counts(normalize=True)}")


    # --- Model Selection and Hyperparameter Tuning ---
    # Define a pipeline that includes the preprocessor and a classifier
    # This ensures that preprocessing steps are applied consistently during cross-validation
    # and when making predictions with the saved model.

    # We will use GridSearchCV to find the best hyperparameters for RandomForestClassifier
    # You can expand this to include other models or a voting classifier.

    # Define the base classifier
    classifier = RandomForestClassifier(random_state=RANDOM_STATE)

    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # This preprocessor is unfitted at this point
        ('classifier', classifier)
    ])
    logging.info("Full pipeline (preprocessor + classifier) created.")

    # Define hyperparameters to tune for RandomForestClassifier
    # This can be expanded to include more parameters or other classifiers
    param_grid = {
        'classifier__n_estimators': [100, 200, 300], # Number of trees in the forest
        'classifier__max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
        'classifier__max_depth': [10, 20, None], # Maximum depth of the tree
        'classifier__min_samples_split': [2, 5], # Minimum number of samples required to split an internal node
        'classifier__min_samples_leaf': [1, 2] # Minimum number of samples required to be at a leaf node
    }
    logging.info(f"Hyperparameter grid for GridSearchCV defined: {param_grid}")

    # Set up GridSearchCV for hyperparameter tuning
    # Using StratifiedKFold to maintain target distribution in each fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        full_pipeline,
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc', # Optimize for ROC AUC score
        n_jobs=-1, # Use all available CPU cores
        verbose=2 # Verbosity level
    )
    logging.info("Starting GridSearchCV for hyperparameter tuning. This may take some time...")

    # Fit GridSearchCV on the training data (X_raw_train, y_train_encoded)
    # The pipeline within GridSearchCV will handle fitting the preprocessor and classifier
    grid_search.fit(X_train, y_train)

    logging.info("GridSearchCV complete.")
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best ROC AUC score on validation sets: {grid_search.best_score_:.4f}")

    # Get the best estimator (the trained pipeline with optimal hyperparameters)
    best_model_pipeline = grid_search.best_estimator_
    logging.info("Retrieved best model pipeline from GridSearchCV.")

    # 5. Evaluate Model
    # Evaluate the best model pipeline on the unseen test data
    evaluate_model(best_model_pipeline, X_test, y_test, target_encoder)

    # 6. Save Model
    # Save the entire pipeline (preprocessor + best classifier)
    save_model(best_model_pipeline, MODEL_PATH)

    logging.info("--- Loan Default Prediction Model Training Script Finished ---")

# Entry point for the script
if __name__ == "__main__":
    # Ensure the data directory exists before trying to load data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info(f"Created data directory: {DATA_DIR}")
        logging.info("Please place 'loan_data.csv' inside the 'data' directory.")
    else:
        logging.info(f"Data directory '{DATA_DIR}' already exists.")

    # Check if the data file exists, if not, provide instructions
    if not os.path.exists(DATA_FILE):
        logging.warning(f"Data file '{DATA_FILE}' not found. Please download 'loan_data.csv' from Kaggle "
                        "and place it in the 'data' directory.")
        logging.warning("You can download it from: https://www.kaggle.com/datasets/ninzaami/loan-prediction-dataset")
        logging.warning("Exiting script. Please add the data file and re-run.")
    else:
        # If data file exists, proceed with training
        main()

