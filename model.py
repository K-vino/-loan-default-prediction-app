# model.py - Comprehensive Loan Default Prediction Model Training Script
# This script is responsible for loading the loan dataset, performing extensive data preprocessing,
# training multiple machine learning models with hyperparameter tuning, evaluating their performance,
# and finally saving the best-performing model pipeline along with its evaluation metrics.
# The goal is to provide a highly robust, well-documented, and production-ready model training pipeline.

# --- Standard Library Imports ---
import pandas as pd # For data manipulation and analysis, especially DataFrames.
import numpy as np  # For numerical operations, especially array manipulation.
import joblib       # For efficient serialization (saving/loading) of Python objects, like scikit-learn models.
import os           # For interacting with the operating system, e.g., managing file paths and directories.
import logging      # For logging events, debugging information, warnings, and errors during script execution.
import json         # For working with JSON data, specifically for saving model evaluation metrics.
from datetime import datetime # For timestamping saved artifacts and log messages.

# --- Scikit-learn Imports for ML Pipeline ---
# Model Selection and Splitting
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# `train_test_split`: Splits arrays or matrices into random train and test subsets.
# `GridSearchCV`: Performs exhaustive search over specified parameter values for an estimator.
# `StratifiedKFold`: Provides train/test indices to split data in train/test sets.
#                    It ensures that the proportion of target classes is preserved in each fold.

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# `LabelEncoder`: Encodes target labels with values between 0 and n_classes-1. Used for the target variable.
# `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
# `OneHotEncoder`: Encodes categorical features as a one-hot numeric array.
from sklearn.impute import SimpleImputer # For handling missing values.
from sklearn.pipeline import Pipeline    # For creating a sequence of data processing steps.
from sklearn.compose import ColumnTransformer # Applies different transformers to different columns of data.

# Classifiers (Machine Learning Models)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# `RandomForestClassifier`: An ensemble meta-estimator that fits a number of decision tree classifiers
#                           on various sub-samples of the dataset and uses averaging to improve the
#                           predictive accuracy and control over-fitting.
# `GradientBoostingClassifier`: Builds an additive model in a forward stage-wise fashion;
#                               it allows for the optimization of arbitrary differentiable loss functions.
# `AdaBoostClassifier`: A meta-estimator that begins by fitting a classifier on the original dataset
#                       and then fits additional copies of the classifier on the same dataset but
#                       where the weights of incorrectly classified instances are adjusted.
from sklearn.linear_model import LogisticRegression # A linear model for binary classification.
from sklearn.svm import SVC # Support Vector Classifier, a powerful and versatile algorithm.

# Model Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# `accuracy_score`: Calculates the accuracy classification score.
# `precision_score`: Calculates the precision of the predictions.
# `recall_score`: Calculates the recall (sensitivity) of the predictions.
# `f1_score`: Calculates the F1 score, which is the harmonic mean of precision and recall.
# `roc_auc_score`: Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# `confusion_matrix`: Computes confusion matrix to evaluate the accuracy of a classification.
# `classification_report`: Builds a text report showing the main classification metrics.

# --- Configuration and Setup ---
# Configure logging for detailed tracking of the model training process.
# Logging messages provide crucial insights into the script's execution flow,
# helping in debugging, monitoring, and understanding the model's behavior.
# `level=logging.INFO`: Sets the minimum level of messages to be displayed (INFO, WARNING, ERROR, CRITICAL).
# `format='%(asctime)s - %(levelname)s - %(message)s'`: Defines the format of log messages,
# including timestamp, log level, and the message itself.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for data, model, and metrics saving.
# Using dedicated directories for different types of artifacts promotes better organization
# and makes the project structure cleaner and easier to manage.
DATA_DIR = "data"       # Directory where the raw dataset (loan_data.csv) is expected to reside.
MODEL_DIR = "models"    # Dedicated directory to store the trained machine learning model pipelines.
METRICS_DIR = "metrics" # Dedicated directory to store model evaluation results and metrics.

# Construct full file paths using os.path.join for cross-platform compatibility.
DATA_FILE = os.path.join(DATA_DIR, "loan_data.csv") # Full path to the input dataset.
MODEL_FILENAME = "loan_model_pipeline.pkl"         # Standard filename for the saved model pipeline.
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME) # Full path where the model will be saved.
METRICS_FILENAME = "model_evaluation_metrics.json" # Standard filename for evaluation metrics.
METRICS_PATH = os.path.join(METRICS_DIR, METRICS_FILENAME) # Full path where metrics will be saved.
TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "target_encoder.pkl") # Path to save the LabelEncoder for the target.
PREPROCESSOR_FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "preprocessor_feature_names.json") # Path to save feature names after preprocessing.

# Define a random state for reproducibility across multiple runs.
# This ensures that data splits (train/test), model initializations, and any
# stochastic processes within algorithms yield the same results, which is vital
# for consistent development, debugging, and validation of the machine learning pipeline.
RANDOM_STATE = 42

# --- Data Loading Function ---
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the loan dataset from the specified CSV file path into a pandas DataFrame.

    This function is designed with robust error handling to gracefully manage common issues
    that might occur during file operations, such as the file not being found or being empty.
    It provides informative log messages to guide the user in case of errors.

    Args:
        file_path (str): The absolute or relative path to the CSV data file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the data file does not exist at the specified `file_path`.
        pd.errors.EmptyDataError: If the data file is found but contains no parsable data.
        Exception: For any other unexpected errors that occur during the file reading process.
    """
    logging.info(f"Attempting to load data from: '{file_path}'...")
    try:
        # Before attempting to read, verify if the file actually exists.
        # This prevents a more generic pandas error and provides a clearer message.
        if not os.path.exists(file_path):
            logging.error(f"Data file does not exist at the specified path: '{file_path}'.")
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Read the CSV file into a pandas DataFrame.
        # `pd.read_csv` is a versatile function for reading delimited data.
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Initial shape: {data.shape} (rows, columns).")
        
        # Log the first few rows of the loaded data to provide a quick visual inspection.
        logging.info(f"First 5 rows of the loaded data:\n{data.head().to_string()}")
        
        # Log concise information about the DataFrame, including column names,
        # non-null counts, and data types, which is essential for initial data quality checks.
        logging.info(f"Data information (df.info()):\n{data.info(verbose=True, show_counts=True)}")
        
        # Log basic descriptive statistics for numerical columns to understand data distribution.
        logging.info(f"Descriptive statistics for numerical columns:\n{data.describe().to_string()}")
        
        return data
    except FileNotFoundError as e:
        logging.critical(f"Critical Error: {e}. Please ensure the data file is correctly placed.")
        raise # Re-raise the exception to signal a fatal error in the script.
    except pd.errors.EmptyDataError:
        logging.critical(f"Critical Error: The CSV file at '{file_path}' is empty or contains no data.")
        raise pd.errors.EmptyDataError("The CSV file is empty.")
    except Exception as e:
        logging.critical(f"An unexpected and critical error occurred during data loading: {e}", exc_info=True)
        raise # Re-raise any other exceptions to prevent silent failures.

# --- Feature Engineering and Preprocessing Pipeline Definition ---
def define_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """
    Defines and returns an unfitted ColumnTransformer for comprehensive data preprocessing.
    This function intelligently identifies numerical and categorical columns from the input DataFrame
    and sets up pipelines for handling missing values, encoding categorical features,
    and scaling numerical features.

    Args:
        df (pd.DataFrame): The raw input DataFrame. This DataFrame is used to infer column types
                           and define the preprocessing steps. It should contain all features
                           that will be used for model training.

    Returns:
        tuple[ColumnTransformer, list, list]:
            - ColumnTransformer: An unfitted `ColumnTransformer` object. This object is designed
                                 to be integrated into a scikit-learn `Pipeline` and will be
                                 fitted on the training data during the `pipeline.fit()` call.
            - list: A list of identified numerical column names.
            - list: A list of identified categorical column names.
    """
    logging.info("Initiating the definition of data preprocessing steps for ColumnTransformer.")
    logging.info("This involves identifying feature types and setting up appropriate transformers.")

    # Create a deep copy of the DataFrame to avoid modifying the original `df`
    # during the process of inferring column types and handling specific features.
    temp_df = df.copy(deep=True)

    # --- Specific Feature Handling: 'Dependents' ---
    # The 'Dependents' column often contains string values like '3+' which need to be
    # converted to a numerical format for machine learning models.
    # We replace '3+' with '3' (or a suitable numerical representation) and then
    # coerce the entire column to a float type. This is crucial for numerical imputation
    # and scaling later in the pipeline.
    if 'Dependents' in temp_df.columns:
        logging.info("Pre-processing 'Dependents' column: converting '3+' to '3' and coercing to numeric (float).")
        # Using .loc to avoid SettingWithCopyWarning and ensure direct modification.
        temp_df.loc[:, 'Dependents'] = temp_df['Dependents'].replace('3+', '3').astype(float)
        # Check for any remaining non-numeric values after replacement and coercion.
        if temp_df['Dependents'].isnull().any():
            logging.warning("NaN values detected in 'Dependents' after initial conversion. Imputation will handle these.")
    else:
        logging.warning("Column 'Dependents' not found in the DataFrame. Skipping specific handling for it.")

    # --- Feature Column Identification ---
    # Define the list of feature columns by excluding 'Loan_ID' and 'Loan_Status'.
    # 'Loan_ID' is a unique identifier and not a predictive feature.
    # 'Loan_Status' is the target variable and should not be treated as a feature.
    feature_cols = [col for col in temp_df.columns if col not in ['Loan_ID', 'Loan_Status']]
    logging.info(f"Identified {len(feature_cols)} feature columns for model training.")

    # Automatically identify numerical and categorical columns from the `feature_cols`.
    # `select_dtypes` is a convenient pandas method for this purpose.
    numerical_cols = temp_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = temp_df[feature_cols].select_dtypes(include=['object', 'bool']).columns.tolist()

    logging.info(f"Automatically identified numerical features: {numerical_cols}")
    logging.info(f"Automatically identified categorical features: {categorical_cols}")

    # --- Numerical Feature Transformation Pipeline ---
    # This pipeline defines the sequence of operations for numerical features:
    # 1. `SimpleImputer(strategy='mean')`: Fills any remaining missing numerical values
    #    with the mean of the respective column. The mean is a robust choice for numerical
    #    imputation when data distribution is not heavily skewed.
    # 2. `StandardScaler()`: Standardizes features by removing the mean and scaling to unit variance.
    #    This is crucial for many machine learning algorithms (e.g., Logistic Regression, SVM,
    #    and even some tree-based methods can benefit) as it prevents features with larger
    #    numerical ranges from disproportionately influencing the model's learning process.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values with the mean.
        ('scaler', StandardScaler())                 # Scale numerical features to a standard range.
    ])
    logging.info("Numerical transformer pipeline defined: SimpleImputer (mean) -> StandardScaler.")

    # --- Categorical Feature Transformation Pipeline ---
    # This pipeline defines the sequence of operations for categorical features:
    # 1. `SimpleImputer(strategy='most_frequent')`: Fills any missing categorical values
    #    with the most frequently occurring category (mode). This is a common and effective
    #    strategy for nominal categorical data.
    # 2. `OneHotEncoder(handle_unknown='ignore')`: Converts categorical variables into a
    #    one-hot encoded numerical format. Each category becomes a new binary column.
    #    `handle_unknown='ignore'` is vital for deployment: if a new, unseen category
    #    appears in the test set or during live prediction, it will be handled gracefully
    #    by assigning zeros to all corresponding one-hot encoded columns, preventing errors.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with the mode.
        ('onehot', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode categorical features.
    ])
    logging.info("Categorical transformer pipeline defined: SimpleImputer (most_frequent) -> OneHotEncoder (handle_unknown='ignore').")

    # --- ColumnTransformer for Parallel Preprocessing ---
    # `ColumnTransformer` is a powerful tool that applies different transformers to different
    # columns of the input data in parallel. This is essential for handling datasets with
    # mixed data types (numerical and categorical) effectively within a single pipeline.
    # - `('num', numerical_transformer, numerical_cols)`: Applies the `numerical_transformer`
    #   pipeline to all columns listed in `numerical_cols`.
    # - `('cat', categorical_transformer, categorical_cols)`: Applies the `categorical_transformer`
    #   pipeline to all columns listed in `categorical_cols`.
    # - `remainder='passthrough'`: This argument ensures that any columns in the input DataFrame
    #   that are *not* explicitly listed in `numerical_cols` or `categorical_cols` are passed
    #   through the transformation process without any modification. In our specific case,
    #   after dropping 'Loan_ID' and separating 'Loan_Status', all remaining features should
    #   ideally be covered by either `numerical_cols` or `categorical_cols`. This acts as a
    #   safety net for unforeseen columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols), # Apply numerical pipeline to numerical columns.
            ('cat', categorical_transformer, categorical_cols) # Apply categorical pipeline to categorical columns.
        ],
        remainder='passthrough' # Pass through any other columns unchanged.
    )
    logging.info("ColumnTransformer created to combine numerical and categorical preprocessing pipelines.")
    logging.info("The preprocessor is now defined and ready to be integrated into the main model pipeline.")
    
    return preprocessor, numerical_cols, categorical_cols

# --- Model Training Function ---
def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Trains various machine learning models with extensive hyperparameter tuning using GridSearchCV.
    The best performing model (based on ROC AUC score from cross-validation) is selected
    and returned as part of a complete scikit-learn pipeline. This pipeline encapsulates
    both the preprocessing steps and the chosen classifier, ensuring consistency.

    Args:
        X_train (pd.DataFrame): The raw training features. This DataFrame should contain
                                the original feature columns before any preprocessing.
                                The `preprocessor` within the pipeline will handle the transformations.
        y_train (pd.Series): The encoded training target variable (e.g., 0s and 1s).
        preprocessor (ColumnTransformer): The unfitted `ColumnTransformer` object defined earlier.
                                          It will be fitted as part of the `full_pipeline.fit()` call.

    Returns:
        Pipeline: The best trained scikit-learn pipeline, which includes the fitted preprocessor
                  and the best-performing classifier with its optimal hyperparameters.
    """
    logging.info("Commencing model training and hyperparameter tuning process.")
    logging.info("Multiple machine learning algorithms will be evaluated to find the optimal model.")

    # Define a dictionary of classifiers and their respective hyperparameter grids for tuning.
    # Each entry specifies a model instance and a dictionary of parameters to search over.
    # The parameter names are prefixed with 'classifier__' because the classifier is
    # a named step within the `full_pipeline` (named 'classifier').
    classifiers = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=2000),
            # `solver='liblinear'`: Good for small datasets and supports L1/L2 penalties.
            # `max_iter`: Increased iterations to ensure convergence for larger C values.
            'params': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], # Inverse of regularization strength. Smaller values specify stronger regularization.
                'classifier__penalty': ['l1', 'l2'] # Specify the norm of the penalty. 'l1' for sparsity, 'l2' for standard regularization.
            },
            'description': "Logistic Regression is a linear model for binary classification, good for interpretability."
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'classifier__n_estimators': [100, 200, 300, 400, 500], # Number of trees in the forest. More trees generally improve performance but increase computation.
                'classifier__max_features': ['sqrt', 'log2', 0.8, None], # Number of features to consider when looking for the best split.
                                                                        # 'sqrt': sqrt(n_features), 'log2': log2(n_features), 0.8: 80% of features, None: all features.
                'classifier__max_depth': [10, 20, 30, 40, None], # Maximum depth of the tree. None means nodes are expanded until all leaves are pure.
                'classifier__min_samples_split': [2, 5, 10, 20], # Minimum number of samples required to split an internal node.
                'classifier__min_samples_leaf': [1, 2, 4, 8], # Minimum number of samples required to be at a leaf node.
                'classifier__bootstrap': [True, False], # Whether bootstrap samples are used when building trees.
                'classifier__class_weight': [None, 'balanced', 'balanced_subsample'] # Weights associated with classes. 'balanced' handles imbalanced datasets.
            },
            'description': "Random Forest is an ensemble method using multiple decision trees, known for its robustness and accuracy."
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'classifier__n_estimators': [100, 200, 300, 400], # Number of boosting stages to perform.
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2], # Shrinks the contribution of each tree.
                'classifier__max_depth': [3, 5, 7, 9], # Maximum depth of the individual regression estimators.
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0], # Fraction of samples to be used for fitting the individual base learners.
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            },
            'description': "Gradient Boosting builds an additive model in a stage-wise fashion, often highly accurate."
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=RANDOM_STATE),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0]
            },
            'description': "AdaBoost is an adaptive boosting algorithm that combines weak learners to form a strong one."
        },
        'SVC': {
            'model': SVC(random_state=RANDOM_STATE, probability=True, cache_size=1000), # probability=True needed for ROC AUC, cache_size for performance.
            'params': {
                'classifier__C': [0.1, 1, 10, 100], # Regularization parameter.
                'classifier__kernel': ['linear', 'rbf', 'poly'], # Specifies the kernel type to be used in the algorithm.
                'classifier__gamma': ['scale', 'auto'] # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            },
            'description': "Support Vector Classifier (SVC) is effective in high-dimensional spaces and for complex decision boundaries."
        }
        # Additional classifiers could be added here following the same structure.
        # Example: 'KNeighborsClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier' (if installed)
    }

    best_model_name = None
    best_roc_auc_score = -np.inf # Initialize with negative infinity to ensure any valid score is better.
    best_estimator_pipeline = None
    
    # Dictionary to store results of each model's GridSearchCV for comparison and logging.
    model_tuning_results = {}

    # Iterate through each defined classifier to perform GridSearchCV.
    for name, config in classifiers.items():
        logging.info(f"\n--- Starting Hyperparameter Tuning for {name} ({config['description']}) ---")
        logging.info(f"Parameters to search: {config['params']}")

        # Create a full pipeline for the current classifier.
        # This pipeline ensures that the preprocessing steps defined in `preprocessor`
        # are applied consistently before the classifier is trained or evaluated.
        # The `preprocessor` object passed here is unfitted; `GridSearchCV` will
        # fit it on each cross-validation fold.
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor), # The unfitted preprocessor ColumnTransformer.
            ('classifier', config['model']) # The current classifier (e.g., RandomForestClassifier).
        ])
        logging.info(f"Pipeline created for {name}: Preprocessor -> {name}.")

        # Define the cross-validation strategy.
        # `StratifiedKFold` is used to ensure that each fold of the cross-validation
        # maintains the same class distribution as the original dataset. This is
        # particularly important for imbalanced datasets like loan default prediction.
        # `n_splits=5`: Divides the data into 5 folds.
        # `shuffle=True`: Shuffles the data before splitting, ensuring randomness.
        # `random_state`: Ensures reproducibility of the folds.
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        logging.info(f"Cross-validation strategy defined: StratifiedKFold with {cv_strategy.n_splits} splits.")

        # Set up GridSearchCV for hyperparameter tuning.
        # `estimator`: The pipeline to be tuned.
        # `param_grid`: The dictionary of hyperparameters to search.
        # `cv`: The cross-validation splitting strategy.
        # `scoring='roc_auc'`: The metric used to evaluate and select the best model.
        #                     ROC AUC is a robust metric for binary classification, especially
        #                     when dealing with imbalanced classes, as it considers both
        #                     true positive rate and false positive rate across all thresholds.
        # `n_jobs=-1`: Instructs GridSearchCV to use all available CPU cores for parallel processing,
        #              significantly speeding up the tuning process.
        # `verbose=2`: Provides a detailed output log during the grid search, showing progress.
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=config['params'],
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
        logging.info(f"GridSearchCV initialized for {name}. Starting fit on training data.")

        # Fit GridSearchCV on the training data.
        # The `fit` method will perform the cross-validation, train models for each
        # combination of hyperparameters, and identify the best set of parameters.
        # It's important to pass the `X_train` (raw features) here, as the `preprocessor`
        # step within the `full_pipeline` will handle the transformations on each fold.
        try:
            grid_search.fit(X_train, y_train)
            logging.info(f"GridSearchCV for {name} completed successfully.")
            logging.info(f"Best parameters found for {name}: {grid_search.best_params_}")
            logging.info(f"Best ROC AUC score on validation sets for {name}: {grid_search.best_score_:.4f}")

            # Store the results for the current model.
            model_tuning_results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_ # Store full CV results for detailed analysis
            }

            # Compare the current model's best score with the overall best found so far.
            if grid_search.best_score_ > best_roc_auc_score:
                best_roc_auc_score = grid_search.best_score_
                best_estimator_pipeline = grid_search.best_estimator_
                best_model_name = name
                logging.info(f"New overall best model found: {best_model_name} with ROC AUC: {best_roc_auc_score:.4f}")
        except Exception as e:
            logging.error(f"An error occurred during GridSearchCV for {name}: {e}", exc_info=True)
            logging.error(f"Skipping {name} due to error.")
            continue # Continue to the next classifier if an error occurs.

    # After iterating through all classifiers, check if a best model was found.
    if best_estimator_pipeline is None:
        logging.critical("No best model could be found after iterating through all classifiers. This indicates a serious issue.")
        raise RuntimeError("No best model found after GridSearchCV. Check data and classifier configurations.")

    logging.info(f"\n--- Overall Best Model Selected: {best_model_name} ---")
    logging.info(f"Best ROC AUC score achieved: {best_roc_auc_score:.4f}")
    logging.info("Model training and hyperparameter tuning process completed.")
    
    return best_estimator_pipeline

# --- Model Evaluation Function ---
def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, target_encoder: LabelEncoder):
    """
    Evaluates the trained machine learning model pipeline on the unseen test dataset.
    This function calculates and logs various performance metrics, providing a comprehensive
    assessment of the model's generalization capabilities. The metrics are also returned
    as a dictionary for persistent storage.

    Args:
        model_pipeline (Pipeline): The fully trained scikit-learn pipeline, including
                                   the fitted preprocessor and the best classifier.
        X_test (pd.DataFrame): The raw test features. The pipeline will apply its
                               preprocessing steps to this data before prediction.
        y_test (pd.Series): The true labels for the test set (encoded numerical values).
        target_encoder (LabelEncoder): The fitted `LabelEncoder` used to transform the
                                       original target labels ('Y'/'N') into numerical
                                       values (1/0). Used here to get original class names
                                       for the classification report.

    Returns:
        dict: A dictionary containing various computed evaluation metrics, suitable for JSON serialization.
    """
    logging.info("Initiating model evaluation on the unseen test set.")

    # Make predictions on the test set.
    # The `model_pipeline.predict()` method automatically orchestrates the
    # preprocessing of `X_test` (using the fitted preprocessor within the pipeline)
    # and then feeds the transformed data to the classifier for prediction.
    y_pred = model_pipeline.predict(X_test)
    logging.info("Predictions generated for the test set.")

    # Predict probabilities for the positive class (class 1, corresponding to 'Approved').
    # `predict_proba()` returns probabilities for all classes; we select the probability
    # for the positive class (index 1). This is necessary for metrics like ROC AUC.
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    logging.info("Probabilities for the positive class generated for the test set.")

    # --- Calculate Core Classification Metrics ---
    # Accuracy: The proportion of correctly classified instances (both positive and negative).
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Calculated Accuracy: {accuracy:.4f}")

    # Precision: Of all instances predicted as positive, what proportion were actually positive.
    # High precision means fewer false positives.
    precision = precision_score(y_test, y_pred)
    logging.info(f"Calculated Precision: {precision:.4f}")

    # Recall (Sensitivity): Of all actual positive instances, what proportion were correctly identified.
    # High recall means fewer false negatives.
    recall = recall_score(y_test, y_pred)
    logging.info(f"Calculated Recall: {recall:.4f}")

    # F1-Score: The harmonic mean of precision and recall. It provides a single metric
    # that balances both precision and recall, useful for imbalanced datasets.
    f1 = f1_score(y_test, y_pred)
    logging.info(f"Calculated F1-Score: {f1:.4f}")

    # ROC AUC Score: Area Under the Receiver Operating Characteristic Curve.
    # This metric measures the ability of the model to distinguish between classes.
    # A higher AUC indicates a better model performance across various classification thresholds.
    # It's particularly robust for imbalanced datasets.
    roc_auc = roc_auc_score(y_test, y_proba)
    logging.info(f"Calculated ROC AUC Score: {roc_auc:.4f}")

    # --- Confusion Matrix ---
    # A table that describes the performance of a classification model on a set of test data
    # for which the true values are known. It shows the number of true positives, true negatives,
    # false positives, and false negatives.
    # Rows represent actual classes, columns represent predicted classes.
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info(f"\nGenerated Confusion Matrix:\n{conf_matrix}")

    # --- Classification Report ---
    # A text report showing the main classification metrics (precision, recall, f1-score, support)
    # for each class, along with overall averages.
    # `target_names`: Provides human-readable labels ('N', 'Y') for the report.
    # `output_dict=True`: Returns the report as a dictionary, which is easier to serialize to JSON.
    class_report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, output_dict=True)
    logging.info(f"\nGenerated Classification Report:\n{json.dumps(class_report, indent=4)}")

    logging.info("Model evaluation process completed successfully.")

    # Compile all calculated metrics into a dictionary.
    # This dictionary will be saved as a JSON file, allowing the Streamlit app
    # to dynamically load and display these performance insights.
    metrics = {
        'timestamp': datetime.now().isoformat(), # Timestamp of when the evaluation was performed.
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc_score': roc_auc,
        'confusion_matrix': conf_matrix.tolist(), # Convert NumPy array to a standard Python list for JSON serialization.
        'classification_report': class_report,
        'best_model_parameters': model_pipeline.named_steps['classifier'].get_params() # Parameters of the best classifier.
    }
    return metrics

# --- Artifact Saving Function ---
def save_artifact(data: any, path: str, artifact_type: str = "model"):
    """
    Saves a given artifact (e.g., a trained scikit-learn model pipeline,
    or a dictionary of evaluation metrics) to a specified file path.
    This function handles creating necessary directories and uses appropriate
    serialization methods (joblib for models, json for metrics).

    Args:
        data (any): The data object to be saved. This can be a scikit-learn Pipeline,
                    a dictionary, or any other serializable Python object.
        path (str): The full file path (including filename) where the artifact should be saved.
        artifact_type (str): A string indicating the type of artifact being saved.
                             Expected values are "model", "metrics", "target_encoder", or "feature_names".
                             This helps in choosing the correct serialization method and logging.
    """
    logging.info(f"Attempting to save {artifact_type} artifact to: '{path}'...")
    try:
        # Ensure that the directory where the artifact will be saved exists.
        # `os.makedirs(exist_ok=True)` creates directories recursively if they don't exist,
        # and does nothing if they already exist, preventing errors.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use different serialization methods based on the artifact type.
        if artifact_type == "model" or artifact_type == "target_encoder":
            # `joblib.dump` is preferred for scikit-learn objects due to its efficiency
            # with large NumPy arrays and compatibility with scikit-learn's internal structures.
            joblib.dump(data, path)
        elif artifact_type == "metrics" or artifact_type == "feature_names":
            # For dictionaries (like metrics or feature names), JSON format is highly readable
            # and universally compatible. `indent=4` makes the JSON output pretty-printed.
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            # Raise a ValueError for unsupported artifact types to prevent incorrect saving.
            logging.error(f"Unsupported artifact type specified: '{artifact_type}'.")
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

        logging.info(f"{artifact_type.capitalize()} artifact successfully saved to: '{path}'.")
    except Exception as e:
        logging.error(f"An error occurred while saving the {artifact_type} artifact to '{path}': {e}", exc_info=True)
        raise # Re-raise the exception to propagate the error.

# --- Main Training Function ---
def main():
    """
    This is the main orchestration function for the entire loan default prediction
    model training pipeline. It encapsulates the sequential execution of all
    major steps: data loading, preprocessing, data splitting, model training
    (including hyperparameter tuning), model evaluation, and saving of all
    relevant artifacts (trained model, evaluation metrics, and utility objects).
    """
    logging.info("--- Starting the main Loan Default Prediction Model Training Script Execution ---")
    logging.info(f"Script initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Ensure all necessary output directories exist.
    # This is a crucial first step to prevent file writing errors later in the script.
    # `exist_ok=True` prevents an error if the directory already exists.
    logging.info("Verifying and creating necessary output directories...")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    logging.info(f"Directories checked/created: '{DATA_DIR}', '{MODEL_DIR}', '{METRICS_DIR}'.")

    # 2. Load the raw dataset.
    # This is a critical step. If data loading fails, the script cannot proceed.
    # Error handling is implemented within `load_data` to provide clear messages.
    try:
        raw_data = load_data(DATA_FILE)
        logging.info("Raw data loaded successfully. Proceeding with data preparation.")
    except (FileNotFoundError, pd.errors.EmptyDataError, Exception) as e:
        logging.critical(f"Aborting script due to critical data loading error: {e}")
        return # Exit the script if data cannot be loaded.

    # Create a working copy of the DataFrame.
    # This practice is good for ensuring that the original `raw_data` DataFrame
    # remains untouched, which can be useful for debugging or re-running parts
    # of the script without re-loading.
    df = raw_data.copy(deep=True)
    logging.info("Created a working copy of the raw data for preprocessing.")

    # 3. Handle 'Loan_ID' column.
    # 'Loan_ID' is a unique identifier and has no predictive power; it should be dropped.
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)
        logging.info("Successfully dropped 'Loan_ID' column from the working DataFrame.")
    else:
        logging.warning("Column 'Loan_ID' not found in the dataset. No action taken for this column.")

    # 4. Separate features (X) and target (y).
    # The target variable 'Loan_Status' is isolated from the predictive features.
    if 'Loan_Status' not in df.columns:
        logging.critical("Target column 'Loan_Status' not found in the dataset. Cannot proceed with model training.")
        return # Exit if the target column is missing.
    X_raw = df.drop('Loan_Status', axis=1) # All columns except 'Loan_Status' are features.
    y_raw = df['Loan_Status']             # 'Loan_Status' is the target.
    logging.info("Features (X_raw) and target (y_raw) successfully separated.")
    logging.info(f"X_raw shape: {X_raw.shape}, y_raw shape: {y_raw.shape}.")

    # 5. Target Encoding.
    # The categorical target variable ('Y'/'N') needs to be converted into numerical
    # format (0s and 1s) for machine learning algorithms.
    # 'Y' (Approved) will be mapped to 1, and 'N' (Not Approved) to 0.
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw)
    logging.info(f"Target variable 'Loan_Status' encoded. Original classes: {target_encoder.classes_} "
                 f"mapped to numerical: {target_encoder.transform(target_encoder.classes_)}.")
    
    # Save the fitted LabelEncoder. This is crucial for decoding predictions back
    # to human-readable labels in the Streamlit app and for consistency.
    save_artifact(target_encoder, TARGET_ENCODER_PATH, artifact_type="target_encoder")
    logging.info(f"Fitted LabelEncoder for target saved to: '{TARGET_ENCODER_PATH}'.")

    # 6. Define Preprocessing Pipeline (ColumnTransformer).
    # This step sets up the blueprint for how numerical and categorical features
    # will be transformed. The `preprocessor` object returned is unfitted at this stage,
    # as it will be fitted during the `train_model` step (within the GridSearchCV's pipeline).
    preprocessor, numerical_cols, categorical_cols = define_preprocessor(X_raw)
    logging.info("Preprocessing ColumnTransformer successfully defined.")
    logging.info(f"Numerical columns identified: {numerical_cols}")
    logging.info(f"Categorical columns identified: {categorical_cols}")

    # Save the feature names that the preprocessor expects and outputs.
    # This is useful for dynamically displaying feature importances in the Streamlit app.
    # We need to fit a temporary preprocessor to get the output feature names.
    try:
        # Create a temporary pipeline just to fit the preprocessor and get feature names
        temp_pipeline_for_features = Pipeline(steps=[('preprocessor', preprocessor)])
        temp_pipeline_for_features.fit(X_raw) # Fit on the entire raw data to get all possible feature names
        preprocessor_feature_names = temp_pipeline_for_features.named_steps['preprocessor'].get_feature_names_out().tolist()
        save_artifact(preprocessor_feature_names, PREPROCESSOR_FEATURE_NAMES_PATH, artifact_type="feature_names")
        logging.info(f"Preprocessor output feature names saved to: '{PREPROCESSOR_FEATURE_NAMES_PATH}'.")
    except Exception as e:
        logging.warning(f"Could not save preprocessor feature names: {e}. This might affect feature importance display.")


    # 7. Split Data into Training and Testing Sets.
    # This split is performed on the raw features (`X_raw`) and encoded target (`y_encoded`).
    # The `test_size` determines the proportion of data allocated for testing.
    # `random_state` ensures the split is reproducible.
    # `stratify=y_encoded` is crucial for maintaining the class distribution of the target
    # variable in both the training and testing sets, especially important for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    logging.info(f"Data split successfully into training (X_train: {X_train.shape}, y_train: {y_train.shape}) "
                 f"and testing (X_test: {X_test.shape}, y_test: {y_test.shape}) sets.")
    logging.info(f"Training set target distribution (0: Not Approved, 1: Approved):\n{pd.Series(y_train).value_counts(normalize=True)}")
    logging.info(f"Testing set target distribution (0: Not Approved, 1: Approved):\n{pd.Series(y_test).value_counts(normalize=True)}")

    # 8. Train Model with Hyperparameter Tuning.
    # This is the core machine learning step. The `train_model` function will
    # perform GridSearchCV across multiple defined classifiers and return the
    # best-performing `Pipeline` object, which includes the fitted preprocessor
    # and the optimized classifier.
    try:
        best_model_pipeline = train_model(X_train, y_train, preprocessor)
        logging.info("Best model pipeline successfully trained and optimized.")
    except RuntimeError as e:
        logging.critical(f"Aborting script due to critical model training error: {e}")
        return # Exit if model training fails.
    except Exception as e:
        logging.critical(f"An unexpected error occurred during model training: {e}", exc_info=True)
        return

    # 9. Evaluate the Best Model.
    # The `evaluate_model` function assesses the performance of the `best_model_pipeline`
    # on the unseen `X_test` data. It calculates and logs various metrics, and returns
    # them in a dictionary.
    evaluation_metrics = evaluate_model(best_model_pipeline, X_test, y_test, target_encoder)
    logging.info("Model evaluation completed. Metrics collected.")

    # 10. Save Model and Metrics.
    # Persist the trained model pipeline and the evaluation metrics to disk.
    # This allows the Streamlit application to load and use the model for predictions
    # and display performance insights without re-training.
    save_artifact(best_model_pipeline, MODEL_PATH, artifact_type="model")
    save_artifact(evaluation_metrics, METRICS_PATH, artifact_type="metrics")
    logging.info("Trained model pipeline and evaluation metrics successfully saved.")

    logging.info(f"--- Loan Default Prediction Model Training Script Finished Successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

# Entry point for the script execution.
# This block ensures that the `main()` function is called only when the script
# is executed directly (not when it's imported as a module).
if __name__ == "__main__":
    # Initial checks to guide the user if data files are missing.
    # This makes the script more user-friendly and robust by providing clear instructions
    # on how to set up the environment before running the training.
    
    # Check if the data directory exists.
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR) # Create the directory if it doesn't exist.
        logging.info(f"Created data directory: '{DATA_DIR}'.")
        logging.warning(f"Data file '{DATA_FILE}' not found. Please download 'loan_data.csv' from Kaggle "
                        "and place it inside the newly created 'data' directory.")
        logging.warning("You can download the dataset from: https://www.kaggle.com/datasets/ninzaami/loan-prediction-dataset")
        logging.warning("Exiting script. Please add the data file and re-run.")
    # If the data directory exists, check if the data file itself is present.
    elif not os.path.exists(DATA_FILE):
        logging.warning(f"Data file '{DATA_FILE}' not found in '{DATA_DIR}'. Please download it from Kaggle "
                        "and place it in the 'data' directory.")
        logging.warning("You can download the dataset from: https://www.kaggle.com/datasets/ninzaami/loan-prediction-dataset")
        logging.warning("Exiting script. Please add the data file and re-run.")
    else:
        # If the data file exists, proceed with the main training process.
        logging.info("Data file found. Initiating model training process.")
        main()

