import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


st.set_page_config(page_title="Deployment", page_icon="🚀")

"""Streamlit app for deployment management."""
st.title('Deployment Management')

# Access the singleton AutoMLSystem instance
automl = AutoMLSystem.get_instance()


# STEP 1: Getting existing artifacts of type pipeline
artifacts = automl.registry.list(type="pipeline")
pipelines = [artifact for artifact in artifacts if artifact.type == 'pipeline']

# Initialize variable to store pipeline

if pipelines:
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline = st.selectbox("Select a Pipeline to Load", options=pipeline_names)

    if st.button('Load Pipeline'):
        # Load the selected artifact
        artifact = next((a for a in pipelines if a.name == selected_pipeline), None)
        if artifact:
            # Deserialize the pipeline from bytes (stored in the artifact's data field)
            try:
                st.session_state.actual_pipeline = pickle.loads(artifact.data)  # Deserialize the pipeline
                
                # Retrieve metadata
                input_features = artifact.metadata.get("input_features", [])
                target_feature = artifact.metadata.get("target_feature", "")
                metrics = artifact.metadata.get("metric_values", {})
                
                # Get the model class name
                model_class_name = st.session_state.actual_pipeline.model.__class__.__name__
                
                # Determine if the model is classification or regression
                if model_class_name in ['MLPClassifierModel',
                                        'KNearestNeighbors',
                                        'DecisionTreeClassifierModel']:
                    model_type = "classification"
                elif model_class_name in ['MultipleLinearRegression', 
                                        'RadiusNeighborsModel',
                                        'LassoModel']:
                    model_type = "regression"
                else:
                    model_type = "Unknown"
                
                st.write("### Pipeline Summary:")
                st.write(f"**Name:** {artifact.name}")
                st.write(f"**Model Type:** {model_type}")
                st.write(f"**Model Used:** {model_class_name}")
                st.write(f"**Input Features:** {[f for f in input_features]}")
                st.write(f"**Target Feature:** {target_feature}")
                st.write("### Metrics and Values:")
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")
                    
            except Exception as e:
                st.error(f"Error loading pipeline: {e}")
        else:
            st.error("Pipeline not found.")
else:
    st.write("No saved pipelines available.")



# STEP 3: Function to generate experiment reports
def generate_experiment_report(predictions, df, model_type, target_feature_name):
    """
    Function to generate experiment reports with metrics and graphs after predictions.
    """
    if model_type == "classification":
        # Plot Confusion Matrix for classification models
        st.write("### Confusion Matrix:")
        if target_feature_name in df.columns:
            y_true = df[target_feature_name]
            # Handle predictions as numpy.ndarray
            y_pred = predictions if isinstance(predictions, np.ndarray) else predictions[:, 0]  # Adjust for ndarray format
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    elif model_type == "regression":
        # For regression models, show actual vs predicted graph
        st.write("### Actual vs Predicted Values (Regression):")
        if target_feature_name in df.columns:
            y_true = df[target_feature_name]
            # For regression, use predictions directly since it's a numpy ndarray
            y_pred = predictions if isinstance(predictions, np.ndarray) else predictions[:, 0]  # Adjust for ndarray format
            fig, ax = plt.subplots()
            ax.scatter(y_true, y_pred)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
            st.pyplot(fig)


# STEP 2: Predictions
st.title('Prediction Management')

# Access the singleton AutoMLSystem instance
automl = AutoMLSystem.get_instance()

st.header("Upload and Predict with New Dataset")

uploaded_file = st.file_uploader("Choose a CSV file for predictions", type='csv')

if uploaded_file is not None:
    # Read the uploaded CSV file into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Get the trained input feature names (excluding the target feature)
    training_input_features = [feature.name for feature in st.session_state.actual_pipeline._input_features]
    target_feature_name = st.session_state.actual_pipeline._target_feature.name
    model_type = st.session_state.actual_pipeline.model.type

    # Remove the target feature column from the uploaded columns (if it exists)
    uploaded_columns = list(df.columns)
    if target_feature_name in uploaded_columns:
        uploaded_columns.remove(target_feature_name)  # Ignore the target feature column

    # Check if the columns in the uploaded dataset match the trained input features
    if set(training_input_features) != set(uploaded_columns):
        st.error("Please upload a dataset with the same column headers as the one used for training. The following columns are missing or misaligned:")
        missing_columns = set(training_input_features) - set(uploaded_columns)
        extra_columns = set(uploaded_columns) - set(training_input_features)
        
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        if extra_columns:
            st.error(f"Extra columns: {', '.join(extra_columns)}")
    else:
        # Convert the DataFrame into a Dataset object
        dataset = Dataset.from_dataframe(df, name="new_data", asset_path="new_data.csv")
        
        # Store the uploaded data in session state to be used within the pipeline
        st.session_state.actual_pipeline.dataset = dataset  # Save the dataset in session_state
        
        # Run predictions if the dataset is uploaded
        if st.button('Run Predictions'):
            try:
                # Run predictions with the loaded pipeline (stored in session_state)
                predictions = st.session_state.actual_pipeline.predict(dataset) 
                st.write("### Predictions:")
                st.dataframe(predictions)
                
                # Call the function to generate experiment reports (metrics and graphs)
                generate_experiment_report(predictions, df, model_type, target_feature_name)

                
            except Exception as e:
                st.error(f"{e}. Please make sure you are uploading a dataset of the same type (classification or regression) as the data you have traineed on.")
