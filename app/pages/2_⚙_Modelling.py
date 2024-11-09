import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
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


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """Write text when needed to inform the user."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section,"
                  "you can design a machine learning "
                  "pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# Step 1: Select a Dataset
dataset_name = st.selectbox('Select a Dataset', [ds.name for ds in datasets])
selected_dataset = next(ds for ds in datasets if ds.name == dataset_name)

# Load the dataset
# Convert the bytes data to a file-like object
data_bytes = selected_dataset.read()  # Assuming this returns bytes
data_file = BytesIO(data_bytes)

# Use pd.read_csv on the file-like object
data = pd.read_csv(data_file)
st.write("Data Preview:", data.head())

# Initialize task_type to None
task_type = None


# Step 2: Feature Detection and Selection
st.write("## Feature Detection")

# Ensure session state attributes are initialized
if 'features' not in st.session_state or st.session_state.features is None:
    st.session_state.features = []
if 'feature_names' not in st.session_state or st.session_state.features is None:
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

# Detect features when the button is pressed
if st.button('Detect Features', key='detect_features'):
    st.session_state.features = detect_feature_types(data_bytes)
    st.session_state.feature_names = [f.name for f in st.session_state.features]

    # Initialize session state attributes based on detected features
    st.session_state.input_features = st.session_state.features[:-1]
    st.session_state.target_feature = st.session_state.features[-1]

    st.session_state.input_features_names = st.session_state.feature_names[:-1]
    st.session_state.target_feature_name = st.session_state.feature_names[-1]

# Allow user to select input and target features
st.session_state.input_features_names = st.multiselect(
    'Select Input Features',
    options=st.session_state.feature_names,
    default=st.session_state.input_features_names
)
st.session_state.target_feature_name = st.selectbox(
    'Select Target Feature',
    options=st.session_state.feature_names,
    index=st.session_state.feature_names.index(st.session_state.target_feature_name) if st.session_state.target_feature_name else 0
)

# Update input and target features based on selection
st.session_state.input_features = [
    f for f in st.session_state.features
    if f.name in st.session_state.input_features_names
]
st.session_state.target_feature = next(
    (f for f in st.session_state.features
     if f.name == st.session_state.target_feature_name),
    None
)

# Detect task type based on selected target feature
if st.session_state.target_feature_name:
    target_feature_data = data[st.session_state.target_feature_name]
    if pd.api.types.is_numeric_dtype(target_feature_data):
        task_type = (
            'regression' if target_feature_data.nunique() > 20
            else 'classification'
        )
    else:
        task_type = 'classification'
    st.success(f"Detected task type: {task_type}")

# Step 3: Model Selection
st.write("## Model Selection")


if task_type == 'classification':
    model_types = ['Decision Tree', 'MLP', 'KNN']
else:
    model_types = ['MLR', 'Lasso', 'Radius Neighbors']


model_type = st.selectbox('Choose Model', model_types)

# Step 4: Model Selection for initialization based on task type
if st.button('Choose Model', key='choose_model'):
    # Create the model based on the selected model type
    if model_type == 'Decision Tree':
        model = DecisionTreeClassifierModel()
    elif model_type == 'MLP':
        model = MLPClassifierModel()
    elif model_type == 'KNN':
        model = KNearestNeighbors()
    elif model_type == 'MLR':
        model = MultipleLinearRegression()
    elif model_type == 'Lasso':
        model = LassoModel()
    elif model_type == 'Radius Neighbors':
        model = RadiusNeighborsModel()
    else:
        st.error("Please select a valid model type.")

    # Store the model in session state to persist through reruns
    st.session_state.model = model
    st.success(f"Model initialized: {model.__class__.__name__}")

# Step 5: Metric Selection
st.write("## Select Metrics")
if 'available_metrics' not in st.session_state:
    st.session_state.features = None
if task_type == 'classification':
    st.session_state.available_metrics = ['Accuracy', 'Average Precision', 'Log Loss']
else:
    st.session_state.available_metrics = ['Mean Squared Error',
                         'R Squared',
                         'Mean Absolute Error']
selected_metrics_names = st.multiselect(
    'Choose Metrics to Evaluate',
    options=st.session_state.available_metrics,
    default=st.session_state.available_metrics[0]
)

# Convert the selected metric names into metrics
selected_metrics = []
for metric in selected_metrics_names:
    selected_metrics.append(get_metric(metric))

# Step 4: Dataset Split with Slider
st.subheader("Data Split Configuration")
split_ratio = st.slider("Set Train/Test Split Ratio",
                        min_value=0.1, max_value=0.9, value=0.8, step=0.05)
st.write(f"Training Data: {split_ratio * 100}%, "
         f"Testing Data: {(1 - split_ratio) * 100}%")

# Step 4: Prepare and split the data
if st.button('Prepare and Split Data', key='prepare_split'):

    st.session_state.pipeline = Pipeline(
        metrics=selected_metrics,
        dataset=selected_dataset,
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

# Step 6: Pipeline Summary
# Check if the pipeline has been created
if 'pipeline' in st.session_state:
    pipeline = st.session_state.pipeline

    # Displaying model type
    st.write("### Model Configuration:")
    st.write(f"**Model Type:** {pipeline.model.__class__.__name__}")

    # Displaying input features
    input_feature_names = [feature.name for feature in st.session_state.input_features]
    st.write(f"**Input Features:** {', '.join(input_feature_names)}")

    # Displaying target feature
    st.write(f"**Target Feature:** {st.session_state.target_feature}")

    # Displaying split ratio
    st.write(f"**Train/Test Split Ratio:** {pipeline._split:.2f}")

    # Displaying selected metrics
    if selected_metrics_names:
        st.write(f"**Selected Metrics:** {', '.join(selected_metrics_names)}")
    else:
        st.write("**Selected Metrics:** None")

    # Add button to execute the model training
    if st.button('Train Model', key='train_model_pipeline'):
        results = st.session_state.pipeline.execute()
        
        st.success("Training completed")
        st.write('### Training Metrics:')
        st.write('**Train Metrics:**')
        for metric, result in results.get('train_metrics'):
            st.write(f"{metric.__class__.__name__}: {result:.5f}")

        st.write('**Test Metrics:**')
        for metric, result in results['test_metrics']:
            st.write(f"{metric.__class__.__name__}: {result:.5f}")

        st.write('### Predictions:')
        st.write('**Train Predictions:**')
        st.write(results['train_predictions'])
        
        st.write('**Test Predictions:**')
        st.write(results['test_predictions'])

else:
    st.warning("Please prepare and split the data before training the model.")


# Step 8: Save Pipeline as an Artifact
st.write("## Save Pipeline")
pipeline_name = st.text_input("Enter Pipeline Name", "MyPipeline")
pipeline_version = st.text_input("Enter Pipeline Version", "1.0")

if st.button('Save Pipeline', key='save_pipeline'):
    if pipeline_name and pipeline_version:
        serialized_pipeline = pickle.dumps(st.session_state.pipeline)
        
        # Get the input and target features for the metadata
        input_features = [f.name for f in st.session_state.input_features] if hasattr(st.session_state, 'input_features') else []
        target_feature = st.session_state.target_feature.name if hasattr(st.session_state, 'target_feature') else ''
        
        artifact = Artifact(
            name=pipeline_name,
            version=pipeline_version,
            data=serialized_pipeline,
            type='pipeline',
            metadata={
            "input_features": input_features,
            "target_feature": target_feature,
            "metrics": selected_metrics
            },
            asset_path=f"pipelines/{pipeline_name}_v{pipeline_version}")
        automl.registry.register(artifact)
        st.success(f"Pipeline '{pipeline_name}' version '{pipeline_version}' saved successfully!")
    else:
        st.error("Please provide both a name and a version for the pipeline.")
else:
    st.warning("Please train the model before saving the Pipeline.")
