import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from io import BytesIO
import pickle

st.set_page_config(page_title="Deployment", page_icon="🚀")

"""Streamlit app for deployment management."""
st.title('Deployment Management')

# Access the singleton AutoMLSystem instance
automl = AutoMLSystem.get_instance()

# Getting existing artifacts of type pipeline
artifacts = automl.registry.list(type="pipeline")
pipelines = [artifact for artifact in artifacts if artifact.type == 'pipeline']

if pipelines:
    pipeline_names = [pipeline.name for pipeline in pipelines]
    selected_pipeline = st.selectbox("Select a Pipeline to Load", options=pipeline_names)

    if st.button('Load Pipeline'):
        # Load the selected artifact
        artifact = next((a for a in pipelines if a.name == selected_pipeline), None)
        if artifact:
            # Deserialize the pipeline from bytes (stored in the artifact's data field)
            try:
                actual_pipeline = pickle.loads(artifact.data)  # Deserialize the pipeline
                
                # Retrieve metadata
                input_features = artifact.metadata.get("input_features", [])
                target_feature = artifact.metadata.get("target_feature", "")
                metrics = artifact.metadata.get("metric_values", {})
                
                st.write("### Pipeline Summary:")
                st.write(f"**Name:** {artifact.name}")
                st.write(f"**Model Type:** {actual_pipeline.model.__class__.__name__}")
                st.write(f"**Input Features:** {[f for f in input_features]}")
                st.write(f"**Target Feature:** {target_feature}")
                st.write("### Metrics and Values:")
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")

                # Step 3: Provide CSV for Predictions
                st.write("### Make Predictions")
                uploaded_file = st.file_uploader("Upload CSV file for predictions", type='csv')

                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of data for prediction:")
                    st.dataframe(df.head())

                    if st.button('Run Predictions'):
                        try:
                            predictions = actual_pipeline.execute(df)  # Assuming the pipeline has an `execute` method
                            st.write("### Predictions:")
                            st.dataframe(predictions)
                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")
            except Exception as e:
                st.error(f"Error loading pipeline: {e}")
        else:
            st.error("Pipeline not found.")
else:
    st.write("No saved pipelines available.")

