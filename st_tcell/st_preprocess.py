from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *


dataset = st.session_state['dataset']
curdataset = st.session_state['curdataset']
config = st.session_state['config']

@st.dialog("View Dateset", width='large')
def viewdataset(dataset):

    col1,col2 = st.columns(2)
    col1.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'><b>Dataset Row</b></h3>",unsafe_allow_html=True)
    col1.markdown(f"<h3 style='display: block;text-align: center;'>{dataset.shape[0]}</h3>",unsafe_allow_html=True)
    col2.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'>Dataset Column</h3>",unsafe_allow_html=True)
    col2.markdown(f"<h3 style='display: block;text-align: center;'>{dataset.shape[1]}</h3>",unsafe_allow_html=True)


    view, info = st.tabs(["Data View", "Dataset Info"])

    with view:
        subset=dataset.select_dtypes(exclude=['object', 'category', 'bool']).columns
        st.dataframe(dataset.style.highlight_max(subset=subset,axis=0,color='Pink').highlight_min(subset=subset,axis=0,color='Aquamarine').highlight_null(color='Linen'))
    with info:
        df = dataset.dtypes.to_frame(name='dtypes')
        df['Non-Null Count'] = dataset.count().transpose()
        df.reset_index(names=['column name'], inplace=True)
        st.dataframe(df, use_container_width=True)



selected_dataset = list(dataset.keys())
container = st.container(border=True)

col0,col1 = container.columns(2)
curdataset = col0.selectbox(
    "Process Dataset:", 
    selected_dataset,
    index = selected_dataset.index(curdataset),
    placeholder = "Select Dataset...",
)
label_col = col1.selectbox(
    "Label Column", 
    dataset[curdataset].columns,
    index=len(dataset[curdataset].columns)-1,
    placeholder="Select label column ...",
)

subcontainer = container.container(border=True)
subcontainer.write("<h3 style='display: block;text-align: center;'>train and test dataset split</h3>", unsafe_allow_html=True)
test_split_rate = subcontainer.slider("test dataset split rate", min_value=0, max_value=100, value=config['preprocess']['test_split_rate'], step=5)
col0,col1,col3 = subcontainer.columns(3)
is_stratify = col0.toggle("Is Stratify", value =config['preprocess']['is_stratify'], help = 'If True, data is split in a stratified fashion, using this as the class labels.')
is_shuffle = col1.toggle("Is Shuffle", value =config['preprocess']['is_shuffle'], help = 'Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.')
is_save_dataset = col3.checkbox("save test dataset", value = True, help = 'If True, save test dataset to disk.')

col0,col1,col2,col3 = container.columns(4)

#subcontainer = container.container(border=True)
col0.write("<h3 style='display: block;text-align: center;'>Rowfilter</h3>", unsafe_allow_html=True)
min_cells_per_type = col0.number_input(
    "Min cells count per T type", min_value=0, value=config['preprocess']['min_cells_per_type'], help='test', placeholder="Type a number..."
)
threshold_null = col0.number_input(
    "Threshold null count per row", min_value=0.0, max_value=1.0, value=config['preprocess']['threshold_null'], step=0.05, placeholder="Type a number..."
)

#subcontainer = container.container(border=True)
col1.write("<h3 style='display: block;text-align: center;'>imputer</h3>", unsafe_allow_html=True)
imputer_list = (None, 'knn', 'linear', 'polynomial', 'decisiontree')
imputermethod = col1.selectbox(
    "Imputer mothod",
    imputer_list,
    index = imputer_list.index(config['preprocess']['imputermethod']),
    placeholder="Select Imputer...",
)
imputer_parm = col1.text_input("Imputer parameter", value="", key="imputer_parm")

#subcontainer = col2.container.container(border=True)
col2.write("<h3 style='display: block;text-align: center;'>Scaler</h3>", unsafe_allow_html=True)
scaler_list = (None,'standard','maxabs','minmax', 'normalizer','power', 'quantile', 'robust')
scalermethod = col2.selectbox(
    "Scaler mothod",
    scaler_list,
    index = scaler_list.index(config['preprocess']['scalermethod']),
    placeholder="Select Scaler...",
)
scaler_parm = col2.text_input("Scaler parameter", value="", key="scaler_parm")

#subcontainer = col3.container.container(border=True)
col3.write("<h3 style='display: block;text-align: center;'>Oversampler</h3>", unsafe_allow_html=True)
oversampler_list = (None,'random', 'smote', 'smoten', 'smotenc', 'borderline', 'adasyn', 'svmsmote', 'kmeans')
oversamplermethod = col3.selectbox(
    "Oversampler mothod",
    oversampler_list,
    index = oversampler_list.index(config['preprocess']['oversamplermethod']),
    placeholder="Select oversampler...",
)
over_parm = col3.text_input("Oversampler parameter", value="", key="over_parm")

col0,col1,col2 = container.columns(3)
viewbtntitle="View Dataset"
click=col0.button(viewbtntitle)
if click:
    viewdataset(dataset[curdataset])

pre=col1.button('Preprocess ...') 
if pre:
    data = dataset[curdataset]
    config['preprocess']['min_cells_per_type'] = min_cells_per_type
    config['preprocess']['threshold_null'] = threshold_null
    config['preprocess']['imputermethod'] = imputermethod
    config['preprocess']['scalermethod'] = scalermethod
    config['preprocess']['test_split_rate'] = test_split_rate
    config['preprocess']['is_stratify'] = is_stratify
    config['preprocess']['oversamplermethod'] = oversamplermethod
    
    data = filter(data, min_cells_per_type=min_cells_per_type, threshold=threshold_null)



    if imputermethod:
        t_type_labels = data.iloc[:, -1].values
        data = impute(data.iloc[:,0:data.columns.size-1], method=imputermethod)
        data[label_col] = t_type_labels
    data_array = data.iloc[:,0:data.columns.size-1].to_numpy()
    feature_list = data.iloc[:,0:data.columns.size-1].keys().values
    t_type_labels = data[label_col].values

    if scalermethod:
        data_array = scale(data_array, scalemethod=scalermethod)
    X_train, X_test, y_train, y_test = train_test_split(data_array, t_type_labels, test_size=test_split_rate/100.0, stratify=t_type_labels if is_stratify else None, random_state=RANDOM_STATE)

    train_data= pd.DataFrame(data = X_train, columns = feature_list)
    test_data= pd.DataFrame(data = X_test, columns = feature_list)
    test_data[label_col] = y_test
    if oversamplermethod:
        if imputermethod:
            train_data = impute(train_data, method = imputermethod, random_state = RANDOM_STATE)
        else:
            train_data = impute(train_data, random_state = RANDOM_STATE)
        train_data,y_train= oversampler(train_data, y_train, method=oversamplermethod)
    train_data[label_col] = y_train
    dataset['train_data'] = train_data
    dataset['test_data'] = test_data
    curdataset = 'train_data'
    if is_save_dataset:
        test_data.to_csv(config['default']['data_path'] / Path('test_data.csv'))
        train_data.to_csv(config['default']['data_path'] / Path('train_data.csv'))
    st.balloons()
st.session_state['dataset'] = dataset
st.session_state['config'] = config

if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = 'Meta_data'
else:
    st.session_state['curdataset'] = curdataset