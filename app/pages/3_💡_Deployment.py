import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from io import BytesIO

st.set_page_config(page_title="Deployment", page_icon="🚀")

def main():
    """Streamlit app for deployment management."""
    st.title('Deployment Management')

    # Access the singleton AutoMLSystem instance
    automl = AutoMLSystem.get_instance()
    
    # Fetching existing artifacts of type pipeline
    artifacts = automl.registry.list(type="artifact")
    pipelines = [artifact for artifact in artifacts if artifact.type == 'pipeline']

    if pipelines:
        pipeline_names = [pipeline.name for pipeline in pipelines]
        selected_pipeline = st.selectbox("Select a Pipeline to Load", options=pipeline_names)

        if st.button('Load Pipeline'):
            # Load the selected artifact
            artifact = next((a for a in pipelines if a.name == selected_pipeline), None)
            if artifact:
                # Convert bytes back to the pipeline object
                actual_pipeline = artifact.data

                st.write("### Pipeline Summary:")
                st.write(f"**Name:** {artifact.name}")
                st.write(f"**Model Type:** {actual_pipeline.model.__class__.__name__}")
                st.write(f"**Input Features:** {[f.name for f in actual_pipeline.input_features]}")
                st.write(f"**Target Feature:** {actual_pipeline.target_feature.name}")
                st.write(f"**Metrics:** {[metric.__class__.__name__ for metric in actual_pipeline.metrics]}")

                # Step 3: Provide CSV for Predictions
                st.write("### Make Predictions")
                uploaded_file = st.file_uploader("Upload CSV file for predictions", type='csv')

                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of data for prediction:")
                    st.dataframe(df.head())

                    if st.button('Run Predictions'):
                        try:
                            predictions = actual_pipeline.execute(df)
                            st.write("### Predictions:")
                            st.dataframe(predictions)
                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Pipeline not found.")
    else:
        st.write("No saved pipelines available.")

if __name__ == "__main__":
    main()
