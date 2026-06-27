import streamlit as st
from src.inference import get_prediction

# Initialise session state variable
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

def app_sidebar():
    st.sidebar.header('Iris Sample Inputs')
    sepal_length = st.sidebar.text_input("Sepal Length (cm)", placeholder="e.g. 5.1")
    sepal_width = st.sidebar.text_input("Sepal Width (cm)", placeholder="e.g. 3.5")
    petal_length = st.sidebar.text_input("Petal Length (cm)", placeholder="e.g. 1.4")
    petal_width = st.sidebar.text_input("Petal Width (cm)", placeholder="e.g. 0.2")

    def get_input_features():
        if not all([sepal_length, sepal_width, petal_length, petal_width]):
            return None
        try:
            return {
                'sepal_length': float(sepal_length),
                'sepal_width': float(sepal_width),
                'petal_length': float(petal_length),
                'petal_width': float(petal_width),
            }
        except ValueError:
            return None
    sdb_col1, sdb_col2 = st.sidebar.columns(2)
    with sdb_col1:
        predict_button = st.sidebar.button("Predict", key="predict")
    with sdb_col2:
        reset_button = st.sidebar.button("Reset", key="clear")
    if predict_button:
        input_features = get_input_features()
        if input_features is None:
            st.sidebar.error("Please enter valid numeric values for all Iris features.")
        else:
            st.session_state['input_features'] = input_features
    if reset_button:
        st.session_state['input_features'] = {}
    return None

def app_body():
    title = '<p style="font-family:arial, sans-serif; color:Black; font-size: 40px;"><b>Iris Setosa Prediction</b></p>'
    st.markdown(title, unsafe_allow_html=True)
    default_msg = '**Prediction result:** {}'
    required_keys = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    if all(key in st.session_state['input_features'] for key in required_keys):
        assessment = get_prediction(
            sepal_length=st.session_state['input_features']['sepal_length'],
            sepal_width=st.session_state['input_features']['sepal_width'],
            petal_length=st.session_state['input_features']['petal_length'],
            petal_width=st.session_state['input_features']['petal_width'],
        )
        if assessment == 1:
            st.success(default_msg.format('Iris Setosa'))
        else:
            st.warning(default_msg.format('Not Iris Setosa'))
    return None

def main():
    app_sidebar()
    app_body()
    return None

if __name__ == "__main__":
    main()