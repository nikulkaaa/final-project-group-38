import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Deployment", page_icon="🚀")
st.title('Deployment Management')

# Access the singleton AutoMLSystem instance
automl = AutoMLSystem.get_instance()


def load_pipeline() -> None:
    """Loads the selected pipeline and retrieves its metadata."""
    artifacts = automl.registry.list(type="pipeline")
    pipelines = ([artifact for artifact
                  in artifacts if artifact.type == 'pipeline'])

    if pipelines:
        pipeline_names = [pipeline.name for pipeline in pipelines]
        selected_pipeline = st.selectbox("Select a Pipeline to Load",
                                         options=pipeline_names)

        if st.button('Load Pipeline'):
            # Load the selected artifact
            artifact = next((
                a for a in pipelines if a.name == selected_pipeline),
                None)
            if artifact:
                try:
                    st.session_state.actual_pipeline = pickle.loads(
                        artifact.data)
                    input_features = artifact.metadata.get("input_features",
                                                           [])
                    target_feature = artifact.metadata.get("target_feature",
                                                           "")
                    metrics = artifact.metadata.get("metric_values", {})
                    actual_model = st.session_state.actual_pipeline.model
                    model_class_name = (
                        actual_model.__class__.__name__
                    )
                    model_type = determine_model_type(model_class_name)

                    st.write("### Pipeline Summary:")
                    st.write(f"**Name:** {artifact.name}")
                    st.write(f"**Model Type:** {model_type}")
                    st.write(f"**Model Used:** {model_class_name}")
                    st.write(f"**Input Features:** {[f for f 
                                                     in input_features]}")
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


def determine_model_type(model_class_name: str) -> str:
    """Determines the model type based on the class name."""
    if model_class_name in ['MLPClassifierModel',
                            'KNearestNeighbors',
                            'DecisionTreeClassifierModel']:
        return "classification"
    elif model_class_name in ['MultipleLinearRegression',
                              'RadiusNeighborsModel',
                              'LassoModel']:
        return "regression"
    return "Unknown"


def generate_experiment_report(predictions: np.ndarray,
                               df: pd.DataFrame,
                               model_type: str,
                               target_feature_name: str) -> None:
    """Creates experiment report (Confusion Matrix / Actual vs Predicted)."""
    if model_type == "classification":
        generate_classification_report(predictions, df, target_feature_name)
    elif model_type == "regression":
        generate_regression_report(predictions, df, target_feature_name)


def generate_classification_report(predictions: np.ndarray,
                                   df: pd.DataFrame,
                                   target_feature_name: str) -> None:
    """Generates confusion matrix for classification models."""
    st.write("### Confusion Matrix:")
    if target_feature_name in df.columns:
        y_true = df[target_feature_name]
        y_pred = predictions if isinstance(predictions,
                                           np.ndarray) else predictions[:, 0]
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)


def generate_regression_report(predictions: np.ndarray,
                               df: pd.DataFrame,
                               target_feature_name: str) -> None:
    """Generates actual vs predicted graph for regression models."""
    st.write("### Actual vs Predicted Values (Regression):")
    if target_feature_name in df.columns:
        y_true = df[target_feature_name]
        y_pred = predictions if isinstance(predictions,
                                           np.ndarray) else predictions[:, 0]
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'k--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)


def what_if_analysis(df: pd.DataFrame,
                     training_input_features: list[str]) -> None:
    """Performs what-if analysis by allowing users to modify input features."""
    st.write("### Modify the Input Features for What-If Analysis:")
    row_index = st.number_input("Select Row to Modify",
                                min_value=0,
                                max_value=len(df) - 1,
                                value=0
                                )
    modified_row = df.iloc[row_index].copy()

    for feature in training_input_features:
        if feature in modified_row:
            min_value = float(df[feature].min())
            max_value = float(df[feature].max())
            default_value = float(df[feature].mean())
            modified_row[feature] = st.slider(
                f"Adjust {feature} (row {row_index})",
                min_value=min_value,
                max_value=max_value,
                value=default_value
            )

    st.write("Modified Row Preview:")
    st.dataframe(modified_row.to_frame().T)

    modified_data = pd.DataFrame(modified_row).T
    modified_dataset = Dataset.from_dataframe(modified_data,
                                              name="modified_data",
                                              asset_path="modified_data.csv")
    st.session_state.actual_pipeline.dataset = modified_dataset

    if st.button('Re-run Predictions with Modified Data'):
        try:
            modified_predictions = st.session_state.actual_pipeline.predict(
                modified_dataset)
            st.write("### Predictions with Modified Data:")
            st.dataframe(modified_predictions)
            model_type = st.session_state.actual_pipeline.model.type
            target_feature_name = (
                st.session_state.actual_pipeline._target_feature.name
            )
            generate_experiment_report(modified_predictions,
                                       modified_data, model_type,
                                       target_feature_name)
        except Exception as e:
            st.error(f"Error during prediction: {e}")


def upload_and_predict() -> None:
    """Handles file upload and prediction."""
    st.header("Upload and Predict with New Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file for predictions",
                                     type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        training_input_features = [
            feature.name for feature
            in st.session_state.actual_pipeline._input_features
        ]
        target_feature_name = (
            st.session_state.actual_pipeline._target_feature.name
        )
        model_type = st.session_state.actual_pipeline.model.type

        uploaded_columns = list(df.columns)
        if target_feature_name in uploaded_columns:
            uploaded_columns.remove(target_feature_name)

        if set(training_input_features) != set(uploaded_columns):
            error = "Please upload a dataset with the same "
            error += "column headers as the one used for training."
            st.error(error)
            missing_columns = set(training_input_features) - set(
                uploaded_columns)
            extra_columns = set(uploaded_columns) - set(
                training_input_features)
            if missing_columns:
                st.error(f"Missing columns: {', '.join(missing_columns)}")
            if extra_columns:
                st.error(f"Extra columns: {', '.join(extra_columns)}")
        else:
            dataset = Dataset.from_dataframe(df,
                                             name="new_data",
                                             asset_path="new_data.csv")
            st.session_state.actual_pipeline.dataset = dataset

            if st.button('Run Predictions'):
                try:
                    predictions = st.session_state.actual_pipeline.predict(
                        dataset
                    )
                    st.write("### Predictions:")
                    st.dataframe(predictions)
                    model_type = st.session_state.actual_pipeline.model.type
                    target_feature_name = (
                        st.session_state.actual_pipeline._target_feature.name
                    )
                    generate_experiment_report(predictions,
                                               df,
                                               model_type,
                                               target_feature_name)
                except Exception as e:
                    error_message = f"{e}. Please ensure the dataset "
                    error_message += "matches the trained model's type."
                    st.error(error_message)

            what_if_analysis(df, training_input_features)


def main() -> None:
    """Main function to run the Streamlit app."""
    load_pipeline()
    upload_and_predict()


if __name__ == "__main__":
    main()
