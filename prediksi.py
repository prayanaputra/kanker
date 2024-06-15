import pickle
import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model
model = pickle.load(open('breast.sav', 'rb'))

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
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.write('The tumor is benign')
    else:
        st.write('The tumor is malignant')

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Kelompok 1 - Machine Learning SI 02</p>
    """,
    unsafe_allow_html=True
)
