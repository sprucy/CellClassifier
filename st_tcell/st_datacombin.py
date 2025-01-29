from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from utils import load_data


dataset = st.session_state['dataset']
curdataset = st.session_state['curdataset']
config = st.session_state['config']

with st.form("data load setup"):
    modality = st.multiselect(
        "Modality",
        ['Ephys', 'Morph'],
        config['dataload']['modality'],
        placeholder="Select Modality...",
    ) 
    taxonomy_list = ['Berg', 'SEA_AD', 'Mouse'] 
    taxonomy = st.selectbox(
        "Taxonomy",
        taxonomy_list,
        index=taxonomy_list.index(config['dataload']['taxonomy']),
        placeholder="Select taxonomy...",
    )
    cell_subset = st.multiselect(
        "Cell Subset",
        ['Superficial','Deep'],
        config['dataload']['cell_subset'],
        placeholder="Select Cell Subset...",
    ) 
    loadsubmit = st.form_submit_button('Generate Dataset...')

if loadsubmit:
    if len(modality) == 0 or len(cell_subset) == 0:
        st.error("Please select at least one modality and one cell subset.")
    else:
        modality_selected = 'Both' if len(modality) == 2 else modality[0]
        cell_subset_selected = 'All' if len(cell_subset) == 2 else cell_subset[0]
        msg = st.toast('Generate data...')
        print(modality_selected, cell_subset_selected)
        labeled_data, unlabeled_data =  load_data(dataset['Meta_data'], dataset['Ephys_data'], dataset['Morph_data'], taxonomy = taxonomy, modality = modality_selected, cell_subset = cell_subset_selected)  
        dataset['Labeled_data'] = labeled_data
        dataset['Unlabeled_data'] = unlabeled_data
        curdataset = 'Labeled_data'
        config['dataload']['taxonomy'] = taxonomy
        config['dataload']['modality'] = modality
        config['dataload']['cell_subset'] = cell_subset
        msg.toast("Generate dataset complete!")
selected_dataset = list(dataset.keys())
container = st.container(border=True)
curdataset = container.selectbox(
    "Current Dataset:", 
    selected_dataset,
    index=selected_dataset.index(curdataset),
    placeholder="Select Dataset...",
)
#container.title(f":red[Current Dataset: {curdataset}]")
col1,col2 = container.columns(2)
col1.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'><b>Dataset Row</b></h3>",unsafe_allow_html=True)
col1.markdown(f"<h3 style='display: block;text-align: center;'>{dataset[curdataset].shape[0]}</h3>",unsafe_allow_html=True)
col2.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'>Dataset Column</h3>",unsafe_allow_html=True)
col2.markdown(f"<h3 style='display: block;text-align: center;'>{dataset[curdataset].shape[1]}</h3>",unsafe_allow_html=True)


view, info = container.tabs(["Data View", "Dataset Info"])

with view:
    subset=dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns
    st.dataframe(dataset[curdataset].style.highlight_max(subset=subset,axis=0,color='Pink').highlight_min(subset=subset,axis=0,color='Aquamarine').highlight_null(color='Linen'))
with info:
    df = dataset[curdataset].dtypes.to_frame(name='dtypes')
    df['Non-Null Count'] = dataset[curdataset].count().transpose()
    df.reset_index(names=['column name'], inplace=True)
    st.dataframe(df, use_container_width=True)

st.session_state['dataset'] = dataset
st.session_state['config'] = config
if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = 'Meta_data'
else:
    st.session_state['curdataset'] = curdataset