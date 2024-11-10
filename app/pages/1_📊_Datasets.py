import streamlit as st
import pandas as pd
from io import BytesIO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


def main() -> None:
    """Streamlit app for managing datasets."""
    st.title('Dataset Management')

    # Access the singleton AutoMLSystem instance
    automl = AutoMLSystem.get_instance()

    # Streamlit sidebar for navigation
    action = st.sidebar.selectbox(
        "Action",
        ["Create Dataset", "View Datasets", "Delete Dataset"]
    )

    if action == "Create Dataset":
        st.header("Upload and Create a New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

        if uploaded_file is not None:
            # Read the uploaded CSV file into DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Converting DataFrame to Dataset Artifact
            if st.button('Create Dataset'):
                dataset = Dataset.from_dataframe(
                    data=df,
                    name=uploaded_file.name,
                    asset_path=uploaded_file.name
                )
                # Save the dataset artifact using the artifact registry
                automl.registry.register(dataset)
                st.success("Dataset created and saved successfully!")

    elif action == "View Datasets":
        st.header("Available Datasets")
        # List all datasets in the artifact registry
        datasets = automl.registry.list(type="dataset")
        if datasets:
            dataset_names = [ds.name for ds in datasets]
            selected_dataset = st.selectbox(
                "Select a dataset to view",
                options=dataset_names
            )
            dataset = next(
                (ds for ds in datasets if ds.name == selected_dataset),
                None
            )
            if dataset:
                st.write("Dataset Content:")
                # Decode the bytes data and read it as a DataFrame
                '''try:'''
                data_bytes = dataset.data
                # Convert bytes to a file-like object and read it as CSV
                df = pd.read_csv(BytesIO(data_bytes))
                st.dataframe(df)
                '''except Exception as e:
                    st.error(f"Error reading dataset: {e}")'''

    elif action == "Delete Dataset":
        st.header("Delete Datasets")
        datasets = automl.registry.list(type="dataset")
        
        if datasets:
            # Map dataset names to their IDs
            dataset_name_to_id = {ds.name: ds.id for ds in datasets}
            dataset_names = list(dataset_name_to_id.keys())
            selected_dataset = st.selectbox(
                "Select a dataset to delete",
                options=dataset_names
            )

            # Button to delete the selected dataset
            if st.button(f"Delete {selected_dataset}"):
                '''try:'''
                # Retrieve the artifact ID of the selected dataset
                artifact_id = dataset_name_to_id.get(selected_dataset)
                if artifact_id is None:
                    st.error("Selected dataset ID not found.")
                else:
                    automl.registry.delete(artifact_id)
                    st.success(f"Dataset '{selected_dataset}' deleted successfully!")
                    
                    # Refresh the datasets list
                    datasets = automl.registry.list(type="dataset")
                    st.experimental_rerun()  # Rerun to refresh the app state
                        
                '''except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error deleting dataset: {e}")'''
if __name__ == "__main__":
    main()
