from pathlib import Path
import streamlit as st
import toml
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from utils import *

# ❓⛳🏅🏆💵📈

@st.dialog("Model Loading", width='large')
def load_modelset():
    config = st.session_state['config']
    dataset = st.session_state['dataset']
    curdataset = st.session_state['curdataset']
    with st.container(border=True):
        model_path = Path(config['default']['model_path'])
        dir_list =[str(dir.relative_to(model_path)) for dir in model_path.iterdir() if dir.is_dir()]
        modelset = st.selectbox(
            "modelset",
            dir_list,
            index = 0,
            placeholder="Select modelset...",
        )
        model_config_file = model_path / modelset / 'model-config.toml'
        if model_config_file.exists():
            model_config = toml.load(model_path / modelset / 'model-config.toml')
            col1,col2 = st.columns(2)
            col1.write({'modality':model_config['dataload']['modality'],'taxonomy':model_config['dataload']['taxonomy'],'cell_subset':model_config['dataload']['cell_subset']})
            col1.write(model_config['preprocess'])
            col2.write(model_config['train'])

    #with st.form(key='modelset_form',border = False): #, clear_on_submit = True):
        model_load_submit = col1.button('Load model')

        if model_load_submit:
            predictor_path = model_path / modelset
            predictor = TabularPredictor.load(predictor_path)
            train_file = model_path / modelset / 'train_data.csv'
            if train_file.exists():
                train_data = pd.read_csv(train_file,index_col=0)
                dataset['train_data'] = train_data
                curdataset = 'train_data'      
                #msg = st.toast("train data loading...")      
            test_file = model_path / modelset / 'test_data.csv'
            if test_file.exists():
                test_data = pd.read_csv(test_file,index_col=0)
                dataset['test_data'] = test_data
                curdataset = 'test_data'

            st.session_state['dataset'] = dataset
            st.session_state['curdataset'] = curdataset
            st.session_state['predictor'] = predictor
            st.session_state['save_path'] = modelset
            st.rerun()

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
            st.rerun()



if 'predictor' in st.session_state:
    predictor = st.session_state['predictor']
    dataset = st.session_state['dataset']
    curdataset = st.session_state['curdataset']
    save_path = st.session_state['save_path']

    selected_dataset = list(dataset.keys())
    with st.container(border=True):

        all_models = predictor.model_names()
        selected_model = predictor.model_best 
        
        select_model_types = st.selectbox(
            "Predict Model",
            all_models,
            index = all_models.index(selected_model),
            placeholder="Select excluded train model ...",
        )
        #included_model_types = container.text_input("Included Model Types", value = predictor.model_best if predictor else 'No model selected') 
        #included_model_types=included_model_types.split('(')[0]
        col10,col11 = st.columns(2, vertical_alignment='bottom')
        predict_dataset = col10.selectbox(
            "Predict Dataset:", 
            selected_dataset,
            index = selected_dataset.index(curdataset),
            placeholder = "Select Predict Dataset...",
        )
        actual_label_col_list = list(dataset[curdataset].columns)
        actual_label_col_list.insert(0,None)
        actual_label_col = col11.selectbox(
            "Actual Label Column", 
            actual_label_col_list,
            index = len(actual_label_col_list)-1,
            placeholder="Select actual label column ...",
        )
        col21,col22,col23,col24 = st.columns([1,2,2,2])
        is_predic = col22.button("Predict")
        is_loaddata = col23.button("Load dataset")
        is_loadmodel = col24.button("Load modelset")
    with st.expander("Models Info", expanded=False):

        event = st.dataframe(predictor.leaderboard(),selection_mode=["multi-row", "single-column"], on_select="rerun",use_container_width=True)
        #event.selection

            #with open(config['default']['model_path'] / Path("SummaryOfModels.html"), 'r', encoding='utf-8') as f:
            #with open(Path("Sweetviz_Report.html"), 'r', encoding='utf-8') as f: 
            #    htmlstr=f.read()   
            #    st.write(htmlstr, unsafe_allow_html=True)
            #    st.html()
    if is_predic:
        with st.expander("Predict result", expanded=True):
            result, proba, leaderboard, confusion,feature_importance = st.tabs(["Predict Result Evaluate", "Predict Probability", "Model Leaderboard", "Confusion Matrix","Feature Importance"])
            with result,st.spinner('Wait compute ...'):
                y_pred = predictor.predict(dataset[predict_dataset].loc[:,dataset[predict_dataset].columns != actual_label_col], model = select_model_types)
                col10,col11 = st.columns(2)
                if actual_label_col:
                    results = pd.concat([y_pred, dataset[predict_dataset][actual_label_col]], axis=1)
                    results.columns=['predicted', 'actual']
                    eval_result = predictor.evaluate(dataset[predict_dataset], model = select_model_types, display=False, detailed_report=True)
                    
                    #col1, col2, col3 = st.columns(3)
                    #col1.metric("Temperature", "70 °F", "1.2 °F")
                    #col2.metric("Wind", "9 mph", "-8%")
                    #col3.metric("Humidity", "86%", "4%")
                    
                    col11.dataframe(pd.DataFrame.from_dict(eval_result['classification_report']).T, use_container_width=True,height=300)

                else:
                    results = pd.DataFrame(y_pred)
                    results.columns=['predicted']
                col10.dataframe(results, use_container_width=True,height=300)

            with proba,st.spinner('Wait compute ...'):
                y_proba = predictor.predict_proba(dataset[predict_dataset], model = select_model_types)
                st.dataframe(y_proba)

            with leaderboard,st.spinner('Wait compute ...'):
                if actual_label_col:
                    # 按测试集分数对所有模型排序
                    models=predictor.leaderboard(dataset[predict_dataset])#,extra_info=True, silent=True,extra_metrics=['accuracy', 'balanced_accuracy','f1_macro','f1_micro', 'f1_weighted'])
                    models.sort_values(by=['score_test','score_val'], ascending=False, inplace=True)
                    subset=models.select_dtypes(exclude=['object', 'category', 'bool']).columns
                    st.dataframe(models.style.highlight_max(subset=subset,axis=0,color='Pink'))
            with confusion,st.spinner('Wait compute ...'):
                if actual_label_col:
                    with st.container(border=True):
                        figtitle = "Confusion Matrix"
                        from sklearn.metrics import confusion_matrix
                        cfm = confusion_matrix(results['actual'], results['predicted'])
                        cfm =cfm[::-1]
                        t_types_updated = list(dataset[predict_dataset][actual_label_col].unique())
                        figtitle = "Confusion Matrix"
                        fig, ax = plt.subplots(figsize=(4, 4))
                        plt.title(figtitle,fontsize=10)
                        im, cbar = heatmap(cfm, t_types_updated[::-1], t_types_updated, ax=ax, cbar_kw={'shrink':0.6}, cmap='YlGn')
                        texts = annotate_heatmap(im, valfmt="{x:.0f}", fontsize=5)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

            with feature_importance,st.spinner('Wait compute ...'):
                if actual_label_col:
                    feature_imp = predictor.feature_importance(dataset[predict_dataset], model = select_model_types)
                    #feature_imp.sort_values(by=['importance'], ascending=False, inplace=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    hbars = ax.barh(feature_imp[feature_imp['importance']!=0].index,feature_imp[feature_imp['importance']!=0]['importance'], align='center')
                    figtitle = 'Feature Importance'
                    ax.bar_label(hbars, feature_imp[feature_imp['importance']!=0]['importance'].round(4), color='white', fontweight='bold', padding=-12, fontsize=3)
                    ax.set_xticklabels(['{:.2f}'.format(a) for a in ax.get_xticks()], fontsize=4)
                    ax.set_yticklabels(feature_imp[feature_imp['importance']!=0].index, fontsize=4)
                    ax.set_xlabel('Importance', fontsize=6)
                    ax.set_ylabel('Feature', fontsize=6)
                    ax.set_title(figtitle, fontsize=10)
                    st.pyplot(plt, use_container_width=True)
                    st.dataframe(feature_imp, use_container_width=True)




    if is_loadmodel:
        load_modelset()
    if is_loaddata:
        load_dataset()
        #st.write(predictor.feature_importance(dataset[predict_dataset].loc[:,dataset[predict_dataset].columns!= actual_label_col]))
        #st.write(predictor.feature_importance(dataset[predict_dataset].loc[:,dataset[predict_dataset].columns!= actual_label_col], model = select_model_types))

    st.session_state['dataset'] = dataset
    if 'curdataset' not in st.session_state:
        st.session_state['curdataset'] = 'Meta_data'
    else:
        st.session_state['curdataset'] = curdataset
else:
    is_load_models = st.toggle("Is load model", value = False if 'predictor' in st.session_state else True, key = 'load_modelset', help = 'If True, load model from disk.')
    if is_load_models:
        load_modelset()

    st.write("No model is available !")
