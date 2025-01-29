import streamlit as st
import pandas as pd
import warnings
from utils import *

# ⛳❓🏅🏆💵📈🥃📤🔀🔃🔄🕎

warnings.filterwarnings("ignore")

if 'predictor' in st.session_state:
    predictor = st.session_state['predictor']
    dataset = st.session_state['dataset']
    curdataset = st.session_state['curdataset']
    save_path = st.session_state['save_path']
    config = st.session_state['config']

    selected_dataset = list(dataset.keys())
    all_models = predictor.model_names()
    selected_model = predictor.model_best 
    
    select_model_types = st.selectbox(
        "Teacher Model",
        all_models,
        index = all_models.index(selected_model),
        placeholder="Select teacher model ...",
    )
    with st.container(border=True):
        col40,col41 = st.columns(2)
        teacher_preds_list = [None,'soft','hard']
        teacher_preds = col40.selectbox(
            'teacher_preds',
            teacher_preds_list,
            index = teacher_preds_list.index(config['train']['teacher_preds']),
            placeholder="Select teacher_preds... [None: only train with original labels (no data augmentation). hard: labels are hard teacher predictions given by teacher.predict(). soft: labels are soft teacher predictions given by teacher.predict_proba()]",

        )
        augment_method_list = [None,'munge','spunge']
        augment_method = col41.selectbox(
            'augment_method',
            augment_method_list,
            index=augment_method_list.index(config['train']['augment_method']),
            placeholder="Select augment_method... [None : no data augmentation performed. munge: The MUNGE algorithm. spunge: A simpler, more efficient variant of the MUNGE algorithm.]",
        ) 
        augmentation_data = col40.selectbox(
            "augmentation data:", 
            selected_dataset,
            index = selected_dataset.index(curdataset),
            placeholder = "Select augmentation_data... An optional extra dataset of unlabeled rows that can be used for augmenting the dataset used to fit student models during distillation (ignored if None)...."
        )
        filte_type_list = [None,'min','25%','mean','50%','75%','max']
        filte_type = col41.selectbox(
            "filte range", 
            filte_type_list,
            index=5,
            placeholder="Select filte range for augmentation data ...",
        )
        is_preserve = col40.checkbox("Is preserve classifications", value=False, help="If True, Classifications with fewer samples are retained.")
        if is_preserve:
            ratio_sample_count = col41.number_input("threshold of sample count", min_value=0.0, max_value=0.3, value=0.15, step=0.01,disabled=False,help="")
        else:
            ratio_sample_count = col41.number_input("threshold of sample count", min_value=0.0, max_value=0.3, value=0.15, step=0.01,disabled=True,help="")
        #ratio_sample_count = col41.slider("threshold of sample count", key="ratio", min_value=0, max_value=30, value=15, step=1)      
          

        
        is_distill = st.button("Distill", type="primary")
        with st.expander("Models Info", expanded=True):

            event = st.dataframe(predictor.leaderboard(),selection_mode=["multi-row", "single-column"], on_select="rerun",use_container_width=True)

        
        if is_distill:
            print(filte_type,is_preserve)
            if filte_type:
                if is_preserve:

                    y_pred = predictor.predict(dataset[augmentation_data].iloc[:,0:-1], model=select_model_types)
                    y_class = y_pred.value_counts(normalize=True)
                    b=dataset[augmentation_data].loc[y_pred[y_pred.isin(y_class[y_class<=ratio_sample_count].index.to_list())].index,:]
                    a=dataset[augmentation_data].loc[y_pred[~y_pred.isin(y_class[y_class<=ratio_sample_count].index.to_list())].index,:]
                    #dataset[augmentation_data][dataset[augmentation_data].iloc[:,-1]!=y_class[y_class<=ratio_sample_count].index.to_list()]
                    y_pred_proba = predictor.predict_proba(a.iloc[:,0:-1], model=select_model_types)
                    idx_filted = y_pred_proba.max(axis=1)[y_pred_proba.max(axis=1)>=y_pred_proba.max(axis=1).describe()[filte_type]].index
                    new_unlabeled_data=pd.concat([dataset[augmentation_data].loc[idx_filted,:],b])
                else:
                    y_pred_proba = predictor.predict_proba(dataset[augmentation_data].iloc[:,0:-1], model=select_model_types)
                    idx_filted = y_pred_proba.max(axis=1)[y_pred_proba.max(axis=1)>=y_pred_proba.max(axis=1).describe()[filte_type]].index
                    new_unlabeled_data=dataset[augmentation_data].loc[idx_filted,:]
            else:
                new_unlabeled_data=dataset[augmentation_data]


            with st.expander("Distilling Log", expanded=True),st.spinner('Wait for Distilling...'):
                with st_stdout("code"), st_stderr("code"):
                    best_model=predictor.model_best
                    predictor.set_model_best(select_model_types)
                    student_models = predictor.distill(augmentation_data = new_unlabeled_data if augmentation_data else None, teacher_preds = teacher_preds, augment_method = augment_method, verbosity = config['train']['verbosity']) 
                    print(predictor.model_best)
                    print(new_unlabeled_data.shape)
            st.session_state['predictor'] = predictor
            st.session_state['dataset'] = dataset
            st.session_state['curdataset'] = curdataset
            st.session_state['save_path'] = save_path
            st.session_state['augmentation_data'] = augmentation_data
            st.session_state['teacher_preds'] = teacher_preds
            st.session_state['augment_method'] = augment_method
            st.balloons()

else:
    is_load_models = st.toggle("Is load model", value = False if 'predictor' in st.session_state else True, key = 'load_modelset', help = 'If True, load model from disk.')
    if is_load_models:
        load_modelset()

    st.write("No model is available !")