import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import KNearestNeighbors
from autoop.core.ml.model.model import DecisionTreeClassifierModel
from autoop.core.ml.model.model import MLPClassifierModel
from autoop.core.ml.model.model import MultipleLinearRegression
from autoop.core.ml.model.model import RadiusNeighborsModel
from autoop.core.ml.model.model import LassoModel
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import get_metric
from io import BytesIO
import pickle


def main() -> None:
    """Streamlit app to manage modelling."""
    # Create the page
    st.set_page_config(page_title="Modelling", page_icon="📈")

    # Inform the user about the modelling page
    st.write("# ⚙ Modelling")
    write_helper_text("In this section,"
                      "you can design a machine learning "
                      "pipeline to train a model on a dataset.")

    # Instantiate the AutoML system
    st.session_state.automl = AutoMLSystem.get_instance()

    # Initialize variables that need initialization
    initialize_needed_variables()

    # Step 1: Select a Dataset
    select_dataset()

    # Step 2: Feature Detection and Selection
    detect_features()

    # Step 3: User selection of target and input features
    select_features()

    # Step 4: Detect task type based on user input
    detect_task_type()

    # Step 5: Model Selection
    choose_model()

    # Step 6: Metric Selection
    select_metrics()

    # Step 7: Prepare and split data
    split_data()

    # Step 8: Pipeline Summary
    generate_pipeline_summary()

    # Step 9: Save Pipeline as an Artifact
    save_pipeline()


def write_helper_text(text: str) -> None:
    """Write text when needed to inform the user."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def select_dataset() -> None:
    """
    Allows the user to select a dataset for modelling.
    Retrieves datasets, enables selection and creates a
    preview of the dataset.
    """
    st.session_state.datasets = (
        st.session_state.automl.registry.list(type="dataset")
    )

    if "datasets" in st.session_state:
        dataset_name = st.selectbox(
            'Select a Dataset',
            [ds.name for ds in st.session_state.datasets]
        )
        st.session_state.selected_dataset = next(
            ds for ds in st.session_state.datasets
            if ds.name == dataset_name
        )

        # Load the dataset
        # Convert the bytes data to a file-like object
        st.session_state.data_bytes = (
            st.session_state.selected_dataset.read()
        )
        data_file = BytesIO(st.session_state.data_bytes)

        # Use pd.read_csv on the file-like object
        st.session_state.data = pd.read_csv(data_file)
        st.write("Data Preview:", st.session_state.data.head())
    else:
        st.warning("Please upload datasets before proceeding.")


def initialize_needed_variables() -> None:
    """Initializenecessary variables."""
    # Initialize task type
    if 'task_type' not in st.session_state:
        st.session_state.task_type = None

    # Ensure session state attributes are initialized
    if 'features' not in st.session_state or (
        st.session_state.features is None
    ):
        st.session_state.features = []
    if 'feature_names' not in st.session_state or (
        st.session_state.features is None
    ):
        st.session_state.feature_names = []

    # Initialize session state attributes if not set by the button
    if 'input_features' not in st.session_state:
        st.session_state.input_features = []
    if 'target_feature' not in st.session_state:
        st.session_state.target_feature = None
    if 'input_features_names' not in st.session_state:
        st.session_state.input_features_names = []
    if 'target_feature_name' not in st.session_state:
        st.session_state.target_feature_name = None


def detect_features() -> None:
    """Detect features in a dataset."""
    st.write("## Feature Detection")

    # Detect features when the button is pressed
    if st.button('Detect Features', key='detect_features'):
        if "datasets" in st.session_state:
            st.session_state.features = detect_feature_types(
                st.session_state.data_bytes
            )
            st.session_state.feature_names = [
                f.name for f in st.session_state.features
            ]

            # Initialize session state attributes based on detected features
            st.session_state.input_features = st.session_state.features[:-1]
            st.session_state.target_feature = st.session_state.features[-1]

            st.session_state.input_features_names = (
                st.session_state.feature_names[:-1]
            )
            st.session_state.target_feature_name = (
                st.session_state.feature_names[-1]
            )
        else:
            st.warning("Upload a dataset first !!")


def select_features() -> None:
    """Select input and target features."""
    # Allow user to select input and target features
    st.session_state.input_features_names = st.multiselect(
        'Select Input Features',
        options=st.session_state.feature_names,
        default=st.session_state.input_features_names
    )
    st.session_state.target_feature_name = st.selectbox(
        'Select Target Feature',
        options=st.session_state.feature_names,
        index=(st.session_state.feature_names.index(
            st.session_state.target_feature_name
        ) if st.session_state.target_feature_name else 0)
    )

    # Update input and target features based on selection
    st.session_state.input_features = [
        f for f in st.session_state.features
        if f.name in st.session_state.input_features_names
    ]
    st.session_state.target_feature = next(
        (f for f in st.session_state.features
         if f.name == st.session_state.target_feature_name), None
    )


def detect_task_type() -> None:
    """Detect the type of task (classification/regression)."""
    # Detect task type based on selected target feature
    if st.session_state.target_feature_name:
        target_feature_data = st.session_state.data[
            st.session_state.target_feature_name
        ]
        if pd.api.types.is_numeric_dtype(target_feature_data):
            st.session_state.task_type = (
                'regression' if target_feature_data.nunique() > 20
                else 'classification'
            )
        else:
            st.session_state.task_type = 'classification'
        st.success(f"Detected task type: {st.session_state.task_type}")


def choose_model() -> None:
    """Choose model for training."""
    st.write("## Model Selection")

    # Determine what models are options acccording to task type
    if st.session_state.task_type == 'classification':
        st.session_state.model_types = ['Decision Tree', 'MLP', 'KNN']
    else:
        st.session_state.model_types = ['MLR', 'Lasso', 'Radius Neighbors']

    # Prompt the user to select one model
    st.session_state.model_type = st.selectbox(
        'Choose Model', st.session_state.model_types
    )

    # Initialize model based on selection
    if st.button('Choose Model', key='choose_model'):
        # Check whether features were created
        if st.session_state.features == []:
            st.warning("Please detect features before initializing model.")
        else:
            # Create the model based on the selected model type
            if st.session_state.model_type == 'Decision Tree':
                model = DecisionTreeClassifierModel()
            elif st.session_state.model_type == 'MLP':
                model = MLPClassifierModel()
            elif st.session_state.model_type == 'KNN':
                model = KNearestNeighbors()
            elif st.session_state.model_type == 'MLR':
                model = MultipleLinearRegression()
            elif st.session_state.model_type == 'Lasso':
                model = LassoModel()
            elif st.session_state.model_type == 'Radius Neighbors':
                model = RadiusNeighborsModel()
            else:
                st.error("Please select a valid model type.")

            # Store the model in session state to persist through reruns
            st.session_state.model = model
            # Inform user about model initialization
            st.success(f"Model initialized: {model.__class__.__name__}")


def select_metrics() -> None:
    """Select metrics to report on the training of the model."""
    st.write("## Select Metrics")
    if 'available_metrics' not in st.session_state:
        st.session_state.features = None
    if st.session_state.task_type == 'classification':
        st.session_state.available_metrics = ['Accuracy',
                                              'Average Precision',
                                              'Log Loss']
    else:
        st.session_state.available_metrics = ['Mean Squared Error',
                                              'Mean Absolute Error']
    st.session_state.selected_metrics_names = st.multiselect(
        'Choose Metrics to Evaluate',
        options=st.session_state.available_metrics,
        default=st.session_state.available_metrics[0]
    )

    # Convert the selected metric names into metrics
    st.session_state.selected_metrics = []
    for metric in st.session_state.selected_metrics_names:
        st.session_state.selected_metrics.append(get_metric(metric))


def split_data() -> None:
    """Allow the user to split the data how they wish. """
    # Create a slider for the user to determine the split
    st.subheader("Data Split Configuration")
    split_ratio = st.slider("Set Train/Test Split Ratio",
                            min_value=0.1, max_value=0.9, value=0.8, step=0.05)
    st.write(f"Training Data: {split_ratio * 100}%, "
             f"Testing Data: {(1 - split_ratio) * 100}%")

    # Step 7: Prepare and split the data
    if st.button('Prepare and Split Data', key='prepare_split'):
        if 'model' in st.session_state:
            st.session_state.pipeline = Pipeline(
                metrics=st.session_state.selected_metrics,
                dataset=st.session_state.selected_dataset,
                model=st.session_state.model,
                input_features=st.session_state.input_features,
                target_feature=st.session_state.target_feature,
                split=split_ratio
            )

            # Preprocess the features
            st.session_state.pipeline._preprocess_features()
            # Apply the split based on the specified ratio
            st.session_state.pipeline._split_data()
            st.success("Data has been prepared and split successfully.")
        else:
            st.warning("Please select model before splitting the data.")


def generate_pipeline_summary() -> None:
    """Train the model and generate pipeline summary."""
    # Check if the pipeline has been created
    if 'pipeline' in st.session_state:
        pipeline = st.session_state.pipeline

        # Displaying model type
        st.write("### Model Configuration:")
        st.write(f"**Model Type:** {pipeline.model.__class__.__name__}")

        # Displaying input features
        input_feature_names = [
            feature.name for feature
            in st.session_state.input_features
        ]
        st.write(f"**Input Features:** {', '.join(input_feature_names)}")

        # Displaying target feature
        st.write(f"**Target Feature:** {st.session_state.target_feature}")

        # Displaying split ratio
        st.write(f"**Train/Test Split Ratio:** {pipeline._split:.2f}")

        # Displaying selected metrics
        if st.session_state.selected_metrics_names:
            st.write(
                "**Selected Metrics:** "
                f"{', '.join(st.session_state.selected_metrics_names)}")
        else:
            st.write("**Selected Metrics:** None")
        train_model()


def train_model() -> None:
    """Train the model on the data from the dataset."""
    # Add button to execute the model training
    if st.button('Train Model', key='train_model_pipeline'):
        st.session_state.results = st.session_state.pipeline.execute()

        st.success("Training completed")
        st.write('### Training Metrics:')
        st.write('**Train Metrics:**')
        for metric, result in st.session_state.results.get('train_metrics'):
            st.write(f"{metric.__class__.__name__}: {result:.5f}")

        st.write('**Test Metrics:**')
        for metric, result in st.session_state.results['test_metrics']:
            st.write(f"{metric.__class__.__name__}: {result:.5f}")

        st.write('### Predictions:')
        st.write('**Train Predictions:**')
        st.write(st.session_state.results['train_predictions'])

        st.write('**Test Predictions:**')
        st.write(st.session_state.results['test_predictions'])


def save_pipeline() -> None:
    """Save a created pipeline."""
    st.write("## Save Pipeline")
    pipeline_name = st.text_input("Enter Pipeline Name", "MyPipeline")
    pipeline_version = st.text_input("Enter Pipeline Version", "1.0")

    if st.button('Save Pipeline', key='save_pipeline'):
        if 'pipeline' not in st.session_state:
            st.warning("Please train the model before saving the Pipeline.")
        else:
            if pipeline_name and pipeline_version:
                serialized_pipeline = pickle.dumps(st.session_state.pipeline)

                # Get the input and target features for the metadata
                input_features = ([f.name
                                  for f in st.session_state.input_features]
                                  if hasattr(st.session_state,
                                             'input_features') else [])
                target_feature = (st.session_state.target_feature.name
                                  if hasattr(st.session_state,
                                             'target_feature') else '')

                # Get the metrics used for evaluation
                metric_values = {}
                for metric, result in (
                    st.session_state.results.get('train_metrics')
                ):
                    metric_name = metric.__class__.__name__
                    metric_values[metric_name] = result

                # Create an arifact with that data
                artifact = Artifact(
                    name=pipeline_name,
                    version=pipeline_version,
                    data=serialized_pipeline,
                    type='pipeline',
                    metadata={
                        "input_features": input_features,
                        "target_feature": target_feature,
                        "metric_values": metric_values,
                    },
                    asset_path=(f"pipelines/{pipeline_name}_"
                                f"v{pipeline_version}"))
                st.session_state.automl.registry.register(artifact)
                st.success(
                    f"Pipeline '{pipeline_name}' version"
                    f" '{pipeline_version}' saved successfully!")
            else:
                st.error("Please provide both a"
                         "name and a version for the pipeline.")


if __name__ == "__main__":
    main()
