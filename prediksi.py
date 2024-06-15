import pickle
import streamlit as st
from PIL import Image

# Load the model
model = pickle.load(open('breast.sav', 'rb'))

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load and display an image
image = Image.open("logo.jpg")
st.image(image, use_column_width=True)

# Define the front end interface
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

st.title('Breast Cancer Prediction')

st.markdown(
    """
    ### Please enter the following details:
    """
)

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    clump_thickness = st.number_input('Clump Thickness', min_value=1, max_value=10, value=1)
    cell_size_uniformity = st.number_input('Cell Size Uniformity', min_value=1, max_value=10, value=1)
    cell_shape_uniformity = st.number_input('Cell Shape Uniformity', min_value=1, max_value=10, value=1)
    marginal_adhesion = st.number_input('Marginal Adhesion', min_value=1, max_value=10, value=1)
    single_epi_cell_size = st.number_input('Single Epithelial Cell Size', min_value=1, max_value=10, value=1)

with col2:
    bare_nuclei = st.number_input('Bare Nuclei', min_value=1, max_value=10, value=1)
    bland_chromatin = st.number_input('Bland Chromatin', min_value=1, max_value=10, value=1)
    normal_nucleoli = st.number_input('Normal Nucleoli', min_value=1, max_value=10, value=1)
    mitoses = st.number_input('Mitoses', min_value=1, max_value=10, value=1)

# Make predictions
if st.button('Predict'):
    input_data = (
        clump_thickness, cell_size_uniformity, cell_shape_uniformity, 
        marginal_adhesion, single_epi_cell_size, bare_nuclei, 
        bland_chromatin, normal_nucleoli, mitoses
    )
    prediction = model.predict([input_data])

    st.markdown(
        """
        ### Prediction Result:
        """
    )

    if prediction[0] == 0:
        st.success('The tumor is **benign**.')
    else:
        st.error('The tumor is **malignant**.')

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">Kelompok 1 - Machine Learning SI 02</p>
    """,
    unsafe_allow_html=True
)
