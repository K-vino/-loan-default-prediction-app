import joblib
import numpy as np
import pandas as pd
import os
import logging

# Configure logging for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the trained model pipeline
MODEL_FILENAME = "loan_model_pipeline.pkl"
MODEL_PATH = os.path.join(".", MODEL_FILENAME) # Assuming model is in the root directory

# Global variable to store the loaded model
_model_pipeline = None

def load_model_pipeline():
    """
    Loads the pre-trained scikit-learn pipeline (including preprocessor and model).
    This function ensures the model is loaded only once.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: For other errors during model loading.
    """
    global _model_pipeline
    if _model_pipeline is None:
        logging.info(f"Attempting to load model pipeline from: {MODEL_PATH}")
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            _model_pipeline = joblib.load(MODEL_PATH)
            logging.info("Model pipeline loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading model: {e}. Please ensure 'loan_model_pipeline.pkl' exists.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during model loading: {e}")
            raise
    return _model_pipeline

def predict_loan_status(input_data: dict) -> tuple[str, float]:
    """
    Predicts whether a loan will be approved based on the provided input features.
    The input data is expected as a dictionary, which will be converted to a DataFrame
    for consistent preprocessing by the loaded pipeline.

    Args:
        input_data (dict): A dictionary containing the borrower's details.
                           Expected keys: 'Gender', 'Married', 'Education', 'Self_Employed',
                           'ApplicantIncome', 'LoanAmount', 'Credit_History'.
                           Values should match the data types expected by the model.

    Returns:
        tuple[str, float]: A tuple containing:
                           - str: The predicted loan status ('Approved ✅' or 'Not Approved ❌').
                           - float: The probability of approval (for the positive class, 'Approved').

    Raises:
        ValueError: If the input_data does not contain all required features.
        Exception: For errors during prediction.
    """
    logging.info(f"Received input for prediction: {input_data}")

    # Define expected features based on the training data
    # These must match the columns used to train the model, excluding 'Loan_ID' and 'Loan_Status'
    expected_features = [
        'Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'LoanAmount', 'Credit_History'
        # Add other features if they were part of the model training and input
        # e.g., 'Dependents', 'Property_Area', 'Loan_Amount_Term'
    ]

    # Validate input_data keys
    if not all(feature in input_data for feature in expected_features):
        missing_features = [f for f in expected_features if f not in input_data]
        logging.error(f"Missing required features in input data: {missing_features}")
        raise ValueError(f"Input data is missing required features: {missing_features}")

    try:
        # Load the model pipeline (it will be loaded only once)
        model_pipeline = load_model_pipeline()

        # Convert the input dictionary to a pandas DataFrame
        # The order of columns matters for the ColumnTransformer in the pipeline
        # Ensure the DataFrame columns match the order and names expected by the preprocessor
        # when it was fitted.
        # It's safer to create a DataFrame with the exact column order.
        input_df = pd.DataFrame([input_data], columns=expected_features)
        logging.info(f"Input DataFrame created:\n{input_df}")

        # Make prediction using the loaded pipeline
        # The pipeline handles all preprocessing (imputation, encoding, scaling) internally
        prediction = model_pipeline.predict(input_df)[0] # Get the first (and only) prediction
        probability = model_pipeline.predict_proba(input_df)[0, 1] # Probability of the positive class (1)

        # Decode the prediction (1 -> Approved, 0 -> Not Approved)
        # Note: The LabelEncoder for the target is not part of the main pipeline,
        # so we manually map 0/1 to 'Not Approved'/'Approved'.
        # In `model.py`, 'Y' was mapped to 1 and 'N' to 0.
        status = 'Approved ✅' if prediction == 1 else 'Not Approved ❌'

        logging.info(f"Prediction: {status}, Probability of Approval: {probability:.4f}")
        return status, probability

    except FileNotFoundError:
        logging.error("Model file not found. Cannot make prediction.")
        return "Error: Model not found", 0.0
    except ValueError as e:
        logging.error(f"Input data error: {e}")
        return f"Error: {e}", 0.0
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}")
        return f"Error: An unexpected error occurred: {e}", 0.0

# Example usage (for testing purposes, not part of the app's direct flow)
if __name__ == "__main__":
    # This block will only run if predictor.py is executed directly
    # In a real Streamlit app, app.py will import and call predict_loan_status

    # Dummy input data for testing
    test_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'LoanAmount': 150,
        'Credit_History': 1.0
    }

    # Ensure a dummy model exists for local testing if model.py hasn't been run
    if not os.path.exists(MODEL_PATH):
        logging.warning(f"Model file '{MODEL_PATH}' not found. Creating a dummy model for testing.")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd

        # Define dummy preprocessor (must match expected features)
        numerical_cols_dummy = ['ApplicantIncome', 'LoanAmount', 'Credit_History']
        categorical_cols_dummy = ['Gender', 'Married', 'Education', 'Self_Employed']

        numerical_transformer_dummy = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer_dummy = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor_dummy = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer_dummy, numerical_cols_dummy),
                ('cat', categorical_transformer_dummy, categorical_cols_dummy)
            ],
            remainder='passthrough'
        )

        # Create a dummy classifier
        classifier_dummy = RandomForestClassifier(random_state=42)

        # Create a dummy pipeline
        dummy_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_dummy),
            ('classifier', classifier_dummy)
        ])

        # To make the dummy pipeline 'fit-able' for joblib, we need to fit it on some dummy data
        # This is a hack for testing predictor.py in isolation without running model.py
        dummy_df = pd.DataFrame([test_input]) # Create a dummy dataframe from test input
        # Add other columns if they are expected by the preprocessor but not in test_input
        # For simplicity, let's just fit on the test_input structure.
        # In a real scenario, you'd load a small sample of your actual data to fit the dummy.
        # For a truly minimal dummy, you might just save a pre-fitted, empty pipeline.

        # To properly fit, we need a target too. Let's make a dummy target.
        dummy_target = pd.Series([1]) # Assume it's approved for dummy fit

        try:
            # Fit the dummy pipeline on dummy data
            dummy_pipeline.fit(dummy_df, dummy_target)
            joblib.dump(dummy_pipeline, MODEL_PATH)
            logging.info("Dummy model pipeline created and saved for testing.")
        except Exception as e:
            logging.error(f"Could not create dummy model for testing: {e}")
            logging.error("Please run model.py first to generate a proper model.")
            exit() # Exit if dummy model cannot be created

    try:
        status, prob = predict_loan_status(test_input)
        print(f"\n--- Test Prediction Result ---")
        print(f"Loan Application Result: {status}")
        print(f"Probability of Approval: {prob:.4f}")
    except Exception as e:
        print(f"\n--- Test Prediction Error ---")
        print(f"An error occurred during test prediction: {e}")

