import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time # For loading indicators
import logging

# Import the prediction logic from predictor.py
from predictor import predict_loan_status

# --- Configuration and Setup ---
# Configure logging for Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Streamlit page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="🏦",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced aesthetics ---
# Using st.markdown with unsafe_allow_html=True for custom styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    .stApp {
        background-color: #f8f9fa; /* Very light gray for a clean background */
    }

    .st-emotion-cache-vk3ypu { /* Main content area padding */
        padding-top: 2.5rem; /* Increased top padding */
        padding-right: 4rem; /* Increased side padding */
        padding-left: 4rem;
        padding-bottom: 2.5rem;
    }

    .st-emotion-cache-1cyp85f { /* Sidebar padding and background */
        padding-top: 2.5rem;
        padding-right: 1.5rem;
        padding-left: 1.5rem;
        padding-bottom: 2.5rem;
        background-color: #e9ecef; /* Slightly darker gray for sidebar */
        border-right: 1px solid #dee2e6; /* Subtle border */
    }

    h1 {
        color: #212529; /* Darker text for main title */
        text-align: center;
        font-weight: 700;
        margin-bottom: 2rem; /* Increased margin */
        border-bottom: 3px solid #007bff; /* Primary blue for accent */
        padding-bottom: 15px; /* Increased padding */
        font-size: 2.8rem; /* Larger font size */
    }

    h2, h3 {
        color: #343a40; /* Dark gray for subheadings */
        font-weight: 600;
        margin-top: 2rem; /* Increased top margin */
        margin-bottom: 1.2rem; /* Increased bottom margin */
        font-size: 1.8rem; /* Larger font size for h2 */
    }

    .stButton>button {
        background-color: #007bff; /* Primary blue button */
        color: white;
        font-weight: 600;
        border-radius: 10px; /* Slightly less rounded for a modern look */
        padding: 0.8rem 1.8rem; /* Increased padding */
        border: none;
        box-shadow: 0 5px 15px rgba(0, 123, 255, 0.3); /* Blue shadow */
        transition: all 0.3s ease;
        width: 100%; /* Make button full width */
        margin-top: 2rem; /* Increased top margin */
        font-size: 1.1rem; /* Larger font size for button text */
    }

    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        box-shadow: 0 8px 20px rgba(0, 123, 255, 0.4); /* More prominent shadow on hover */
        transform: translateY(-3px); /* More pronounced lift effect */
    }

    .stTextInput>div>div>input,
    .stSelectbox>div>div>div>div,
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ced4da; /* Light gray border */
        padding: 0.7rem 1.2rem; /* Increased padding */
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.08); /* Subtle inner shadow */
        background-color: white; /* White background for inputs */
    }

    .stAlert {
        border-radius: 8px;
        font-size: 1rem;
        padding: 1rem 1.5rem;
    }

    .stSuccess {
        background-color: #d4edda; /* Light green for success */
        color: #155724; /* Dark green text */
        border-left: 5px solid #28a745; /* Bootstrap success green */
        padding: 1.2rem; /* Increased padding */
        border-radius: 8px;
        margin-top: 1.5rem; /* Increased margin */
        font-weight: 500;
    }

    .stWarning {
        background-color: #fff3cd; /* Light yellow for warning */
        color: #856404; /* Dark yellow text */
        border-left: 5px solid #ffc107; /* Bootstrap warning yellow */
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-weight: 500;
    }

    .stError {
        background-color: #f8d7da; /* Light red for error */
        color: #721c24; /* Dark red text */
        border-left: 5px solid #dc3545; /* Bootstrap danger red */
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-weight: 500;
    }

    .metric-box {
        background-color: white;
        border-radius: 12px;
        padding: 2rem; /* Increased padding */
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1); /* More pronounced shadow */
        text-align: center;
        margin-bottom: 1.5rem; /* Increased margin */
        border: 1px solid #e9ecef; /* Subtle border */
    }
    .metric-box h3 {
        color: #007bff; /* Primary blue for metric titles */
        margin-top: 0;
        margin-bottom: 0.8rem; /* Increased margin */
        font-size: 1.5rem;
    }
    .metric-box p {
        font-size: 3rem; /* Larger font size for metric values */
        font-weight: 700;
        color: #343a40; /* Dark gray for metric values */
        margin: 0;
    }

    /* Adjust Streamlit's default radio button/selectbox styling for better look */
    .stRadio > label, .stSelectbox > label {
        font-weight: 600;
        color: #495057; /* Slightly darker label text */
        margin-bottom: 0.5rem;
    }

    /* Horizontal rule styling */
    hr {
        border-top: 2px solid #adb5bd; /* Darker, more visible separator */
        margin-top: 3rem;
        margin-bottom: 3rem;
    }

    /* Info box styling */
    .stAlert.stInfo {
        background-color: #e2f2ff; /* Lighter blue for info */
        color: #004085; /* Darker blue text */
        border-left: 5px solid #007bff;
    }

    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("---") # Add a separator in sidebar
page_selection = st.sidebar.radio(
    "Go to",
    ("Loan Prediction", "About This App", "Model Insights", "Data Overview")
)
st.sidebar.markdown("---") # Add another separator

# --- Helper Function for Input Validation ---
def validate_inputs(inputs: dict) -> tuple[bool, str]:
    """
    Performs basic validation on the user inputs.

    Args:
        inputs (dict): Dictionary of user inputs, including all UI fields.

    Returns:
        tuple[bool, str]: A tuple where the first element is True if inputs are valid,
                          False otherwise. The second element is an error message if invalid.
    """
    # Validate numerical inputs
    if inputs['ApplicantIncome'] <= 0:
        return False, "Applicant Income must be greater than 0."
    if inputs['LoanAmount'] <= 0:
        return False, "Loan Amount must be greater than 0."
    if inputs['Loan_Amount_Term'] <= 0:
        return False, "Loan Amount Term must be greater than 0."
    if inputs['Dependents'] < 0: # This is now guaranteed to be an int
        return False, "Number of Dependents cannot be negative."

    # Validate categorical/binary inputs
    if inputs['Credit_History'] not in [0.0, 1.0]:
        return False, "Credit History must be 0.0 (Bad) or 1.0 (Good)."

    # Add more complex validation rules as needed
    if inputs['LoanAmount'] > inputs['ApplicantIncome'] * 10 and inputs['ApplicantIncome'] > 0:
        st.warning("Warning: Loan amount is significantly higher than applicant income. This might impact approval.")
    if inputs['CoapplicantIncome'] < 0:
        return False, "Coapplicant Income cannot be negative."

    return True, ""

# --- Page Content Functions ---

def loan_prediction_page():
    """
    Displays the main loan prediction form and results.
    """
    st.title("🏦 Loan Default Prediction App")
    st.write("Fill in the details below to predict the loan approval status.")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Select the applicant's gender."
        )
        married = st.selectbox(
            "Married",
            ["Yes", "No"],
            help="Is the applicant married?"
        )
        # Changed: Dependents will now be an integer directly for validation
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10, # Max value for reasonable input
            value=0,
            step=1,
            help="Number of dependents the applicant has (e.g., 0, 1, 2, 3)."
        )
        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"],
            help="Applicant's education level."
        )
        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"],
            help="Is the applicant self-employed?"
        )

    with col2:
        st.subheader("Loan Details")
        applicant_income = st.number_input(
            "Applicant Income ($)",
            min_value=0,
            value=5000,
            step=100,
            help="Applicant's monthly income."
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income ($)",
            min_value=0,
            value=0,
            step=100,
            help="Coapplicant's monthly income (if any)."
        )
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=0,
            value=150000, # Changed default to a more realistic value for dollars
            step=1000,
            help="Loan amount requested in dollars."
        )
        loan_amount_term = st.selectbox(
            "Loan Amount Term (Months)",
            [12, 36, 60, 120, 180, 240, 300, 360, 480],
            index=7, # Default to 360 months (30 years)
            help="Term of the loan in months."
        )
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)",
            help="Does the applicant have a good credit history (1.0) or not (0.0)?"
        )
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Rural", "Semiurban"],
            help="Location of the property."
        )

    st.markdown("---") # Separator

    # Prepare input dictionary for validation and potential model input
    # All UI inputs are collected here.
    input_features = {
        'Gender': gender,
        'Married': married,
        'Dependents': int(dependents), # Ensure Dependents is an integer
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': float(applicant_income),
        'CoapplicantIncome': float(coapplicant_income),
        'LoanAmount': float(loan_amount),
        'Loan_Amount_Term': float(loan_amount_term),
        'Credit_History': float(credit_history),
        'Property_Area': property_area
    }

    # Features that the current `predict_loan_status` function (and the trained model) expects:
    # IMPORTANT: If you update model.py to include more features like Dependents, CoapplicantIncome,
    # Loan_Amount_Term, Property_Area, you MUST add them to this list and re-train your model.
    predictor_expected_features = [
        'Gender', 'Married', 'Education', 'Self_Employed',
        'ApplicantIncome', 'LoanAmount', 'Credit_History'
    ]

    # Create the dictionary that will actually be passed to the predictor.
    # This ensures only features the model was trained on are sent.
    actual_input_for_predictor = {
        key: input_features[key] for key in predictor_expected_features
    }

    if st.button("Predict Loan Status"):
        # Validate all UI inputs first
        is_valid, validation_message = validate_inputs(input_features)

        if not is_valid:
            st.error(f"Input Error: {validation_message}")
        else:
            with st.spinner("Predicting loan status..."):
                time.sleep(1) # Simulate a short delay for prediction
                try:
                    # Call the prediction function from predictor.py
                    result_status, result_proba = predict_loan_status(actual_input_for_predictor)

                    st.success(f"Loan Application Result: **{result_status}**")
                    st.info(f"Probability of Approval: **{result_proba:.2f}**")

                    if result_status == 'Approved ✅':
                        st.balloons() # Fun animation for approval
                        st.markdown("Great news! Your loan application is likely to be approved.")
                    else:
                        st.warning("Your loan application is likely to be **Not Approved**. "
                                   "Consider reviewing your details or credit history.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("Please ensure the model file `loan_model_pipeline.pkl` exists and is correctly trained "
                             "and that its expected features match the ones provided.")
                    logging.exception("Error during prediction in Streamlit app.")

    st.markdown("---")
    st.subheader("How this prediction works:")
    st.info("""
        This application uses a Machine Learning model (Random Forest Classifier) trained on historical loan data.
        It analyzes factors like your income, credit history, and other demographic details to predict
        the likelihood of your loan being approved or defaulted.

        **Note**: While the UI collects more details (like Dependents, Coapplicant Income, Loan Term, Property Area),
        the current prediction model is configured to use only Gender, Married, Education, Self-Employed,
        Applicant Income, Loan Amount, and Credit History. To include the additional features in the prediction,
        you would need to update and re-train the model (`model.py`) to incorporate them.
        """)

def about_page():
    """
    Displays information about the application.
    """
    st.title("About the Loan Default Prediction App")
    st.markdown("""
    This application is designed to demonstrate a machine learning workflow, from data preparation
    and model training to deployment using Streamlit.

    ### 🚀 Objective
    The primary goal of this project is to predict whether a borrower will default on a loan.
    This kind of prediction is crucial for financial institutions to assess risk and make informed
    decisions regarding loan approvals.

    ### 🛠️ Technologies Used
    * **Python**: The core programming language.
    * **scikit-learn**: For machine learning model training (Random Forest, Logistic Regression, etc.),
        preprocessing (scaling, encoding, imputation), and pipeline management.
    * **pandas**: For data manipulation and analysis.
    * **numpy**: For numerical operations.
    * **Streamlit**: For creating the interactive web application user interface.
    * **joblib**: For saving and loading the trained machine learning model.
    * **matplotlib & seaborn**: For data visualization.

    ### 🧠 Model Details
    The model used is a **Random Forest Classifier**. This is an ensemble learning method
    that operates by constructing a multitude of decision trees during training and
    outputting the class that is the mode of the classes (classification) or mean prediction (regression)
    of the individual trees. It's known for its robustness and good performance.

    The model is trained on a dataset containing various applicant features such as:
    * Gender, Marital Status, Education, Self-Employment Status
    * Applicant Income, Loan Amount, Credit History
    * (Potentially) Number of Dependents, Coapplicant Income, Loan Term, Property Area

    ### 📈 How it Works
    1.  **Data Loading**: The application loads a pre-defined loan dataset.
    2.  **Preprocessing**: The raw data undergoes several preprocessing steps:
        * **Missing Value Imputation**: Filling in missing data points (e.g., with mean for numerical, mode for categorical).
        * **Categorical Encoding**: Converting text-based categorical features into numerical representations (e.g., One-Hot Encoding).
        * **Numerical Scaling**: Scaling numerical features to a standard range to prevent features with larger values from dominating the model.
    3.  **Model Training**: A Random Forest Classifier is trained on the preprocessed data.
        Hyperparameter tuning (using GridSearchCV) is performed to find the optimal settings for the model.
    4.  **Prediction**: When a user inputs new data, it goes through the exact same preprocessing steps
        (ensured by a scikit-learn Pipeline) and then the trained model makes a prediction.

    ### ☁️ Deployment
    This app is designed to be easily deployable using platforms like Streamlit Sharing,
    which allows you to host your Streamlit applications directly from a GitHub repository.

    ---
    *Disclaimer: This app is for educational and demonstration purposes only and should not be used for real-world financial decision-making.*
    """)

def model_insights_page():
    """
    Provides some basic insights into the model, like feature importance (if available).
    """
    st.title("Model Insights")
    st.write("Understanding what drives the model's predictions.")

    st.markdown("""
    ### Feature Importance
    For tree-based models like Random Forest, we can often extract feature importances,
    which indicate how much each feature contributes to the model's predictions.

    **Note**: To display accurate feature importances, the `loan_model_pipeline.pkl`
    must contain a fitted `RandomForestClassifier` and the preprocessor should
    retain original feature names or provide a way to map them back after one-hot encoding.
    Due to the dynamic nature of `ColumnTransformer` and `OneHotEncoder`,
    getting exact feature names after transformation can be complex without
    explicitly storing them during `model.py` training.

    For demonstration, we will use a placeholder or a simplified approach.
    """)

    # Attempt to load the model and extract feature importance
    try:
        # Access the globally loaded model from predictor.py if it exists
        model_pipeline = predict_loan_status.__globals__.get('_model_pipeline')
        if model_pipeline is None:
            model_pipeline = predict_loan_status.load_model_pipeline() # Load if not already loaded

        classifier = model_pipeline.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            st.subheader("Top Features by Importance")
            # Get feature names after preprocessing
            # This is a simplified approach; a robust solution needs to handle OneHotEncoder output names
            # For now, let's use the `predictor_expected_features` as a base for feature names.
            # In a real scenario, you'd get these from the fitted preprocessor using `get_feature_names_out()`.

            # Define feature names that the model was trained on (from predictor_expected_features)
            # and expand for one-hot encoded categorical features if possible.
            # This requires knowing the categories the OneHotEncoder learned during training.
            # For a more accurate display, you'd need to save the preprocessor's feature names or categories.

            # As a placeholder, let's use the `predictor_expected_features` and add common one-hot encoded suffixes.
            # This is an approximation for visualization.
            base_features = [
                'ApplicantIncome', 'LoanAmount', 'Credit_History',
                'Gender', 'Married', 'Education', 'Self_Employed'
            ]
            # These are the *original* features the model was trained on.
            # The actual feature_importances_ array corresponds to the features *after* preprocessing.
            # To correctly map, we need the feature names from the fitted ColumnTransformer.

            # A robust way to get feature names after ColumnTransformer:
            try:
                preprocessor_step = model_pipeline.named_steps['preprocessor']
                # This method is available in scikit-learn 0.23+
                feature_names_out = preprocessor_step.get_feature_names_out()
                feature_importances = pd.Series(classifier.feature_importances_, index=feature_names_out)
                feature_importances = feature_importances.sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax, palette='viridis')
                ax.set_title("Feature Importances (Random Forest)")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
            except AttributeError:
                st.warning("Could not retrieve exact feature names from the preprocessor. "
                           "Displaying raw feature importances. For better labels, ensure scikit-learn version 0.23+ "
                           "and that the preprocessor's `get_feature_names_out()` method works as expected.")
                st.write("Raw Feature Importances:")
                st.write(classifier.feature_importances_)
            except Exception as e:
                st.error(f"Error getting feature names from preprocessor: {e}")
                st.write("Raw Feature Importances:")
                st.write(classifier.feature_importances_)


        else:
            st.info("Feature importances are not available for the current model type or it's not a tree-based model.")

    except Exception as e:
        st.error(f"Could not load model or extract insights: {e}")
        st.warning("Please ensure `model.py` has been run successfully to create `loan_model_pipeline.pkl`.")
        logging.exception("Error loading model for insights page.")

    st.markdown("---")
    st.subheader("Model Performance Overview")
    st.info("""
    During training, the model's performance was evaluated using metrics like Accuracy, Precision, Recall, F1-Score, and ROC AUC.
    These metrics help us understand how well the model generalizes to unseen data.

    * **Accuracy**: Overall correctness of predictions.
    * **Precision**: Of all positive predictions, how many were actually correct.
    * **Recall**: Of all actual positive cases, how many were correctly identified.
    * **F1-Score**: Harmonic mean of precision and recall.
    * **ROC AUC**: Measures the ability of the model to distinguish between classes. A higher AUC indicates a better model.
    """)

def data_overview_page():
    """
    Provides a basic overview of the dataset used for training.
    """
    st.title("Dataset Overview")
    st.write("A glimpse into the data that powers our prediction model.")

    st.markdown("""
    The model was trained on the **Loan Prediction Dataset** from Kaggle.
    This dataset contains various attributes of loan applicants and their corresponding loan status.
    """)

    st.subheader("Sample Data")
    # To avoid loading the entire dataset into the app's memory,
    # we'll create a small dummy DataFrame that resembles the actual data structure.
    # In a production app, you might load a small sample or metadata.
    try:
        # Attempt to load a small sample of the actual data if available
        data_path = os.path.join("data", "loan_data.csv")
        if os.path.exists(data_path):
            sample_df = pd.read_csv(data_path).head(10) # Load first 10 rows
            st.dataframe(sample_df)
            st.write(f"Displaying first {len(sample_df)} rows of the dataset.")

            st.subheader("Data Distribution Insights")
            st.markdown("""
            Below are some basic visualizations showing the distribution of key features
            and their relationship with the target variable (Loan Status).
            """)

            # Plot 1: Loan Status Distribution
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Loan_Status', data=sample_df, palette='viridis', ax=ax1)
            ax1.set_title('Loan Status Distribution')
            ax1.set_xlabel('Loan Status')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)

            # Plot 2: Applicant Income Distribution
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(sample_df['ApplicantIncome'], kde=True, bins=30, color='skyblue', ax=ax2)
            ax2.set_title('Applicant Income Distribution')
            ax2.set_xlabel('Applicant Income')
            ax2.set_ylabel('Frequency')
            st.pyplot(fig2)

            # Plot 3: Loan Amount Distribution
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.histplot(sample_df['LoanAmount'], kde=True, bins=30, color='lightcoral', ax=ax3)
            ax3.set_title('Loan Amount Distribution')
            ax3.set_xlabel('Loan Amount')
            ax3.set_ylabel('Frequency')
            st.pyplot(fig3)

            # Plot 4: Credit History vs. Loan Status
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            sns.countplot(x='Credit_History', hue='Loan_Status', data=sample_df, palette='pastel', ax=ax4)
            ax4.set_title('Loan Status by Credit History')
            ax4.set_xlabel('Credit History (0=Bad, 1=Good)')
            ax4.set_ylabel('Count')
            st.pyplot(fig4)

        else:
            st.warning(f"Data file '{data_path}' not found. Cannot display sample data or insights.")
            st.info("Please ensure `loan_data.csv` is in the `data/` directory.")

    except Exception as e:
        st.error(f"An error occurred while trying to load or display sample data: {e}")
        logging.exception("Error loading sample data for overview page.")

    st.markdown("---")
    st.subheader("Dataset Features")
    st.markdown("""
    The dataset includes the following features:
    * `Loan_ID`: Unique Loan ID (identifier, not used for prediction)
    * `Gender`: Male/Female
    * `Married`: Yes/No
    * `Dependents`: Number of dependents (0, 1, 2, 3+)
    * `Education`: Graduate/Not Graduate
    * `Self_Employed`: Yes/No
    * `ApplicantIncome`: Applicant income
    * `CoapplicantIncome`: Coapplicant income
    * `LoanAmount`: Loan amount in thousands
    * `Loan_Amount_Term`: Term of loan in months
    * `Credit_History`: Credit history meets guidelines (1.0 for yes, 0.0 for no)
    * `Property_Area`: Urban/Rural/Semiurban
    * `Loan_Status`: Loan approved (Y/N) - **Target Variable**
    """)


# --- Main App Logic (Page Routing) ---
if __name__ == "__main__":
    if page_selection == "Loan Prediction":
        loan_prediction_page()
    elif page_selection == "About This App":
        about_page()
    elif page_selection == "Model Insights":
        model_insights_page()
    elif page_selection == "Data Overview":
        data_overview_page()

