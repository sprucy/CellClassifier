
from pathlib import Path
import streamlit as st
import toml
from autogluon.tabular import TabularDataset, TabularPredictor

@st.dialog("Model Info", width='large')
def viewpredictor():
    model_path = Path("model")
    dir_list =[str(dir.relative_to(model_path)) for dir in model_path.iterdir() if dir.is_dir()]
    modelset = st.selectbox(
        "modelset",
        dir_list,
        index = 0,
        placeholder="Select modelset...",
    )
    model_config = toml.load(model_path / modelset / 'model-config.toml')
    model_load_submit = st.button('load models ...')
    st.write(model_config)

    if model_load_submit:
        predictor_path = model_path / modelset
        predictor = TabularPredictor.load(predictor_path)
        st.session_state['predictor'] = predictor
        st.rerun()

if st.button('is load models...'):
    viewpredictor()

if 'predictor' in st.session_state:
    predictor = st.session_state['predictor']
    st.write(predictor.leaderboard())

#st.json(model_config)