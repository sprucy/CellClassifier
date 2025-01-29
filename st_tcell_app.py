from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import toml
from st_pages import add_page_title, get_nav_from_toml, hide_pages
from utils import INDEX_COLUMN

st.set_page_config(
    page_title="Cell Type Prediction App",
    page_icon="🍪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is an Cell Type Prediction App!"
    }
)

st.logo("images/Nywq5flIWE.jpg")
nav = get_nav_from_toml(".streamlit\st-pages.toml")
config = toml.load(".streamlit\st-config.toml")

pages=[]
for _, value in nav.items():
    pages+=value
pagesname=[page.title for page in pages]
dataset = {}
# Define the path to the files and read in the data
data_path = Path(config['default']['data_path'])
meta_path = data_path / config['dataload']['meta_file']             # read the meta data file
ephys_path = data_path / config['dataload']['ephys_file']           # read the ephys data file
morph_path = data_path / config['dataload']['morph_file']           # read the morph data file
# Read in the data files
msg = st.toast("Raw data loading...")
meta_data = pd.read_csv(meta_path, index_col=0,dtype={'ephys_session_id':str}) 
ephys_data = pd.read_csv(ephys_path, index_col=0)
morph_data = pd.read_csv(morph_path, index_col=0) 
dataset['Meta_data'] = meta_data
dataset['Ephys_data'] = ephys_data
dataset['Morph_data'] = morph_data
curdataset = 'Meta_data'
msg.toast('Raw data loaded!')

if 'config' not in st.session_state:
    st.session_state['config'] = config
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = dataset
if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = curdataset
if 'page' not in st.session_state:
    st.session_state['pages'] = pagesname

pg = st.navigation(nav)
add_page_title(pg)
pg.run()

