import pickle
import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model
try:
    model = pickle.load(open('breast.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'breast.sav' is in the correct location.")
    st.stop()

# Load and display an image
image = Image.open("logo.jpg")
st.image(image, use_column_width=True)

# Define the front end interface
st.title('Breast Cancer Prediction')

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ### Please enter the following details:
    """
)

# Get user input
clump_thickness = st.number_input('Clump Thickness')
cell_size_uniformity = st.number_input('Cell Size Uniformity')
cell_shape_uniformity = st.number_input('Cell Shape Uniformity')
marginal_adhesion = st.number_input('Marginal Adhesion')
single_epi_cell_size = st.number_input('Single Epithelial Cell Size')
bare_nuclei = st.number_input('Bare Nuclei')
bland_chromatin = st.number_input('Bland Chromatin')
normal_nucleoli = st.number_input('Normal Nucleoli')
mitoses = st.number_input('Mitoses')

if st.button('Predict'):
    input_data = [[clump_thickness, cell_size_uniformity, cell_shape_uniformity, marginal_adhesion, single_epi_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]]
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.write('The tumor is benign')
        else:
            st.write('The tumor is malignant')
    except NotFittedError:
        st.error("Model is not fitted. Please train the model before making predictions.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Load the dataset for accuracy calculation
try:
    data = pd.read_csv('breastcancer.csv')  # Correct dataset name
    
    st.write("Dataset loaded. Shape before processing:", data.shape)
    st.write(data.head())

    # Check for non-numeric values and missing values
    st.write("Checking for non-numeric values and missing values...")

    non_numeric = data.applymap(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
    if non_numeric.any().any():
        st.write("Non-numeric values found in the dataset:")
        for column in data.columns:
            if non_numeric[column].any():
                st.write(f"Column '{column}' contains non-numeric values.")
                st.write(data.loc[non_numeric[column], column])

    missing_values = data.isna().sum()
    if missing_values.any():
        st.write("Missing values found in the dataset:")
        st.write(missing_values[missing_values > 0])

    # Ensure data is numeric and handle missing values
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    
    st.write("Processed dataset. Shape after processing:", data.shape)
    st.write(data.head())
    
    # Check if data is empty after processing
    if data.empty:
        st.error("The processed dataset is empty. Please check your data.")
    else:
        # Assuming the last column is the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Make predictions on the dataset
        predictions = model.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)

        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

except FileNotFoundError:
    st.error("Dataset file not found. Please ensure 'breastcancer.csv' is in the correct location.")
except Exception as e:
    st.error(f"An error occurred while calculating accuracy: {e}")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Kelompok 1 - Machine Learning SI 02</p>
    """,
    unsafe_allow_html=True
)
