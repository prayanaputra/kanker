import pickle
import streamlit as st

# Load the model
model = pickle.load(open('breast.sav', 'rb'))

# Define the front end interface
st.title('Breast Cancer Prediction')

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

# Make predictions
if st.button('Predict'):
    input_data = (clump_thickness, cell_size_uniformity, cell_shape_uniformity, marginal_adhesion, single_epi_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses)
    prediction = model.predict([input_data])

    if prediction[0] == 0:
        st.write('The tumor is benign')
    else:
        st.write('The tumor is malignant')
