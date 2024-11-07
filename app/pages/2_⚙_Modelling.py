import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.model import KNearestNeighbors
from autoop.core.ml.model.model import DecisionTreeClassifierModel
from autoop.core.ml.model.model import MLPClassifierModel
from autoop.core.ml.model.model import MultipleLinearRegression
from autoop.core.ml.model.model import RadiusNeighborsModel
from autoop.core.ml.model.model import LassoModel
from autoop.core.ml.pipeline import Pipeline
from io import BytesIO


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

"""
Things to take care of:
- Select Model based on Model Type (done?)
- Make a check for the task type from the data (continuous vs categorical)
- Take care of creating a specific model
and fitting it instead of training it on specific features

Regression Models: Multiple Linear Regression, Lasso, Radius Neighbors
Classification Models: KNN, MLP, Decision Tree
"""

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
if st.button('Detect Features', key='detect_features'):
    features = detect_feature_types(data_bytes)
    feature_names = [f.name for f in features]

    # Initialize session state attributes if they don't exist
    if 'input_features' not in st.session_state:
        st.session_state.input_features = features[:-1]
    if 'target_feature' not in st.session_state:
        st.session_state.target_feature = features[-1]
    if 'input_features_names' not in st.session_state:
        st.session_state.input_features_names = feature_names[:-1]
    if 'target_feature_name' not in st.session_state:
        st.session_state.target_feature_name = feature_names[-1]

    # Allow user to select input and target features
    st.session_state.input_features_names = st.multiselect(
        'Select Input Features',
        options=feature_names,
        default=st.session_state.input_features_names
    )
    st.session_state.target_feature_name = st.selectbox(
        'Select Target Feature',
        options=feature_names,
        index=feature_names.index(st.session_state.target_feature_name)
    )

    # Update input and target features based on selection
    st.session_state.input_features = (
        [f for f in features
         if f.name in st.session_state.input_features_names]
    )
    st.session_state.target_feature = next(
        (f for f in features
         if f.name == st.session_state.target_feature_name),
        None)

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


# Step 4: Dataset Split with Slider
st.subheader("Data Split Configuration")
split_ratio = st.slider("Set Train/Test Split Ratio",
                        min_value=0.1, max_value=0.9, value=0.8, step=0.05)
st.write(f"Training Data: {split_ratio * 100}%, "
         f"Testing Data: {(1 - split_ratio) * 100}%")

# Step 4: Prepare and split the data
if st.button('Prepare and Split Data', key='prepare_split'):
    st.session_state.pipeline = Pipeline(
        metrics=[],
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


# Step 5: Metric Selection
st.write("## Select Metrics")
if task_type == 'classification':
    available_metrics = ['Accuracy', 'Average Precision', 'Log Laws']
else:
    available_metrics = ['Mean Squared Error',
                         'R squared',
                         'Mean Absolute Error']
selected_metrics = st.multiselect(
    'Choose Metrics to Evaluate',
    options=available_metrics,
    default=available_metrics[0]
)


# Step 6: Pipeline Summary and Execution
st.write("## Pipeline Summary")
if st.button('Train Model', key='train_model_pipeline'):
    results = st.session_state.model.fit(
        st.session_state.pipeline.train_X,
        st.session_state.pipeline.train_y
    )
    st.success("Training completed")
    st.write('### Results: ')
    st.write(results)

# Observations: When you call _split data in pipeline: train_X
# Ground truth: train_Y
