from pathlib import Path
import streamlit as st
import toml
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils import *

# ❓⛳🏅🏆💵📈

@st.dialog("Dataset Loading", width='large')
def load_dataset():
    config = st.session_state['config']
    dataset = st.session_state['dataset']
    curdataset = st.session_state['curdataset']
    data_path = Path(config['default']['data_path'])
    pattern = '*.csv'
    file_list = [str(file.stem) for file in data_path.glob(pattern) if file.is_file()]
    file_list.insert(0,None)
    data_file = st.selectbox(
        "data file",
        file_list,
        index = 0,
        placeholder="Select data file...",
    )
    if data_file and st.button('Load data'):
        data_file = (data_path / data_file).with_suffix(pattern[1:])
        df = pd.read_csv(data_file,index_col=0)
        dataset[str(data_file.stem)] = df
        st.session_state['dataset'] = dataset
        st.session_state['curdataset'] = str(data_file.stem)
        st.rerun()

    is_uploaded_datafile = st.toggle("Is upload data file", value = False, key = 'load_dataset', help = 'If True, load data from file.')

    if is_uploaded_datafile:
        uploaded_file = st.file_uploader("Choose a CSV, XLSX, and PKL file",  type=['csv','xlsx','pkl'])
        if uploaded_file is not None:
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file,index_col=0)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.pkl'):
                df = pd.read_pickle(uploaded_file)
            else:
                raise ValueError("Invalid file type. Only CSV, XLSX, and PKL files are supported.")
            df.to_csv(data_path / uploaded_file.name)
            dataset[str(Path(uploaded_file.name).stem)] = df
            st.session_state['dataset'] = dataset
            st.session_state['curdataset'] = str(Path(uploaded_file.name).stem)
            st.rerun()

dataset = st.session_state['dataset']
curdataset = st.session_state['curdataset']

#dataset_key = 'edit_dataset'                            # value_key of data_editor
#dataset_editor_key = data_editor_key(dataset_key)      # editor_key of data_editor
#if dataset_key not in st.session_state:           # initialize session_state.value_key
#    st.session_state[dataset_key] = dataset[curdataset]

#print(dataset[curdataset])

selected_dataset = tuple(dataset.keys())
container = st.container(border=True)

curdataset = container.selectbox(
    "Current Dataset:", 
    selected_dataset,
    index=selected_dataset.index(curdataset),
    placeholder="Select Dataset...",
)
label_col = container.selectbox(
    "Label Column", 
    dataset[curdataset].columns,
    index=len(dataset[curdataset].columns)-1,
    placeholder="Select label column ...",
)

col21,col22,col23,col24 = container.columns([1,2,2,2])
is_loaddata = col23.button("Load dataset")

container.divider() 

col1,col2 = container.columns(2)
col1.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'><b>Dataset Row</b></h3>",unsafe_allow_html=True)
col1.markdown(f"<h3 style='display: block;text-align: center;'>{dataset[curdataset].shape[0]}</h3>",unsafe_allow_html=True)
col2.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'>Dataset Column</h3>",unsafe_allow_html=True)
col2.markdown(f"<h3 style='display: block;text-align: center;'>{dataset[curdataset].shape[1]}</h3>",unsafe_allow_html=True)


view, info = container.tabs(["Data View", "Dataset Info"])

with view:
    is_edit_dataset = st.toggle("Is edit dataset", value = False, key = '_is_edit_dataset', help = 'If True, enabled edit dataset.')
    if is_edit_dataset:
        #edit_dataset =st.data_editor(st.session_state[dataset_key].copy(), num_rows="dynamic", key=dataset_editor_key,args=(dataset_key, dataset_editor_key))
        dataset[curdataset] =st.data_editor(dataset[curdataset].copy(), hide_index = True, num_rows="dynamic", key = '_dataset_editor_key')
    else:
        subset=dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns
        event =st.dataframe(dataset[curdataset].style.highlight_max(subset=subset,axis=0,color='Pink')
                 .highlight_min(subset=subset,axis=0,color='Aquamarine')
                 .highlight_null(color='Linen'),
                 selection_mode=["multi-row", "single-column"], on_select="rerun")

with info:
    df = dataset[curdataset].dtypes.to_frame(name='dtypes')
    df['Non-Null Count'] = dataset[curdataset].count().transpose()
    df.reset_index(names=['column name'], inplace=True)
    st.dataframe(df, use_container_width=True)
    


if is_loaddata:
    load_dataset()
    #st.write(predictor.feature_importance(dataset[predict_dataset].loc[:,dataset[predict_dataset].columns!= actual_label_col]))
    #st.write(predictor.feature_importance(dataset[predict_dataset].loc[:,dataset[predict_dataset].columns!= actual_label_col], model = select_model_types))

#dataset[curdataset] = edit_dataset
st.session_state['dataset'] = dataset
if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = 'Meta_data'
else:
    st.session_state['curdataset'] = curdataset

