import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from sidebar_body import text as side_body
from sidebar_intro import text as side_intro

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
from training import utils

# Paths
saved_models_path = os.path.abspath('pre-trained')
example_files_path = os.path.join(base_path, "app/examples")


columns_capacity = ['Voltage', 'Current', 'Voltage Average', 'Current Average', 'Temperature', 'Capacity']
columns = ['Voltage', 'Current', 'Voltage Average', 'Current Average', 'Temperature']
lstm_steps = 300


# Helper Functions
def list_saved_models(path):
    """List all pre-trained models."""
    if not os.path.exists(path):
        st.error(f"Path not found: {path}")
        return []
    return [model for model in os.listdir(path) if not model.startswith('.') and model.startswith('comb')]


def list_example_files(path):
    """List all example files."""
    if not os.path.exists(path):
        st.error(f"Path not found: {path}")
        return []
    return [example for example in os.listdir(path) if not example.startswith('.')]


def load_model(model_path):
    """Load TensorFlow model."""
    try:
        full_path = os.path.join(saved_models_path, model_path)
        model = tf.keras.models.load_model(full_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Main Application
def main() -> None:
    st.header("Predict Battery State of Charge :battery: :bar_chart:")
    st.sidebar.markdown(side_intro, unsafe_allow_html=True)
    st.sidebar.subheader(":gear: App Options")

    # Sidebar Options
    models = list_saved_models(saved_models_path)
    examples = list_example_files(example_files_path)

    if not models:
        st.sidebar.error("No pre-trained models found.")
        st.stop()
    if not examples:
        st.sidebar.error("No example files found.")
        st.stop()

    selected_model = st.sidebar.selectbox("Select pre-trained model", models)
    example_file = st.sidebar.selectbox("Select example file", examples)
    example_file_path = os.path.join(example_files_path, example_file)
    st.sidebar.markdown(side_body, unsafe_allow_html=True)

    resample_1hz = True
    vi_averages = True
    is_lstm = "lstm" in selected_model

    # Getting Started Section
    with st.expander("Getting Started"):
        st.write(Path("app/getstarted.md").read_text())

    # Upload File Section
    st.subheader("Upload cell discharge cycle CSV file")
    uploaded_data = st.file_uploader("Drag and Drop or Click to Upload", type=".csv")
    if uploaded_data is None:
        st.info(f"Using example: {example_file}")
        uploaded_data = example_file_path
    else:
        st.success("Uploaded your file!")

    # Load Dataset
    dataset, dataset_norm = utils.app_create_dataset(uploaded_data, vi_averages, resample_1hz)
    if dataset.empty:
        st.error(f"Failed to read file: {uploaded_data}", icon="🚨")
        st.stop()

    # Processed Data Exploration
    with st.expander("Explore Processed Data"):
        st.markdown("### Chart")
        fig, axes = plt.subplots(5, figsize=(10, 15))
        for i, col in enumerate(columns):
            dataset[col][:100000].plot(ax=axes[i], legend=True)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("### Statistics")
        st.write(dataset.describe().transpose())

    # Normalized Data Exploration
    with st.expander("Explore Normalized Data"):
        st.markdown("### Chart")
        fig, axes = plt.subplots(5, figsize=(10, 15))
        for i, col in enumerate(columns):
            dataset_norm[col][:100000].plot(ax=axes[i], legend=True)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("### Statistics")
        st.write(dataset_norm.describe().transpose())

    # Prediction Section
    st.subheader(f"Evaluating {uploaded_data} with model {selected_model}")
    with st.spinner("Running Predictions..."):
        if "Capacity" in dataset_norm.columns:
            if is_lstm:
                test_features, test_labels = utils.create_lstm_dataset(dataset_norm, lstm_steps)
                test_labels = utils.keep_only_y_end(test_labels, lstm_steps)
            else:
                test_features = dataset_norm[columns_capacity].copy()
                test_labels = test_features.pop("Capacity")

            model = load_model(selected_model)
            if not model:
                st.error("Failed to load the model.")
                return

            evaluation = model.evaluate(test_features, test_labels, verbose=2)
            st.table(pd.DataFrame({"Metric": model.metrics_names, "Result": evaluation}).T)

            predictions = model.predict(test_features).flatten()
            fig, ax = plt.subplots()
            ax.plot(predictions[:1000], label="Predicted SOC")
            if is_lstm:
                ax.plot(test_labels[:1000], label="True SOC")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Capacity column not found. Skipping evaluation.")


if __name__ == "__main__":
    st.set_page_config("SOC ML App", "🔋", layout="wide")
    main()
