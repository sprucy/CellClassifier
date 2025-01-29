import streamlit as st
from st_pages import hide_pages
from autogluon.tabular import TabularDataset, TabularPredictor
import sys
import datetime
from pathlib import Path
from contextlib import contextmanager
from io import StringIO
import toml
import warnings

from utils import *

warnings.filterwarnings("ignore")

@st.dialog("Model Info", width='large')
def viewpredictor(predictor):
    st.write("AutoGluon infers problem type is: ", predictor.problem_type)
    st.write("AutoGluon identified the following types of features:")
    st.write(predictor.feature_metadata)
    st.write("Best Model:", predictor.model_best)
    result = predictor.fit_summary(show_plot=False)
    st.write(result)


config = st.session_state['config']
dataset = st.session_state['dataset']
curdataset = st.session_state['curdataset']

selected_dataset = list(dataset.keys())
container = st.container(border=True)

models_list = ['GBM','CAT','XGB','RF','XT','KNN','LR','NN_TORCH','FASTAI']
models_note_list = ['(LightGBM)','(CatBoost)','(XGBoost)','(Random Forest)','(Extremely Randomized Trees)','(K-Nearest Neighbors)','(Linear Regression)','(Neural Network For Pytorch)','(Neural Network with FastAI)']
#[x + y for x, y in zip(models_list, models_note_list)],
#[x + y for x, y in zip(models_list, models_note_list)],

excluded_model_types = container.multiselect(
    "excluded Model",
    models_list,
    config['train']['excluded_model_types'],
    placeholder="Select excluded train model ...",
) 
#included_model_types=included_model_types.split('(')[0]
col10,col11 = container.columns(2)
curdataset = col10.selectbox(
    "Train Dataset:", 
    selected_dataset,
    index = selected_dataset.index(curdataset),
    placeholder = "Select Dataset...",
)
label_col = col11.selectbox(
    "Label Column", 
    dataset[curdataset].columns,
    index=len(dataset[curdataset].columns)-1,
    placeholder="Select label column ...",
)

verbosity = col10.slider("verbosity", min_value=0, max_value=3, value=config['train']['verbosity'], step=1)
time_limit = col11.slider("time_limit", min_value=0, max_value=3600, value=config['train']['time_limit'], step=60)

#设置训练精度
presets_list=['Customize', 'medium_quality', 'good_quality','high_quality', 'best_quality', 'optimize_for_deployment', 'interpretable', 'ignore_text']
presets = col10.selectbox(
    "presets", 
    presets_list,
    index = presets_list.index(config['train']['presets']),
    placeholder="Select presets ...",
)
metric_list = [None, 'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted','log_loss', 'pac_score' 
               'precision', 'average_precision', 'precision_macro', 'precision_micro', 'precision_weighted',
               'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'roc_auc','roc_auc_ovo_macro',
               'r2', 'mean_absolute_error','mean_squared_error', 'root_mean_squared_error', 'median_absolute_error', 
               'mean_absolute_percentage_error', 'symmetric_mean_absolute_percentage_error',
               'mae', 'mse', 'mape','msle','rmsle','smape','spearmanr',]
eval_metric = col11.selectbox(
    "metric",
    metric_list,
    index = metric_list.index(config['train']['eval_metric']),
    placeholder="Select metric...",
)

if presets == 'Customize':
    auto_stack = container.toggle("Auto Stack", value = config['train']['auto_stack'], 
                                help = '''Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy. 
                                Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy! Automatically sets 
                                num_bag_folds and num_stack_levels arguments based on dataset properties. Note: Setting num_bag_folds and num_stack_levels arguments 
                                will override auto_stack. Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.''')

    if auto_stack:
        with container.expander(":rainbow[☰ Auto Stack Args ☰]", expanded=True):
            col20,col21 = container.columns(2)
            # bagging的折数,缺省好像未8，暂未设置，可在fit中设置
            num_bag_folds = col20.slider("num_bag_folds", min_value=0, max_value=8, value=config['train']['num_bag_folds'], step=1)
            # stacking的级别，暂未设置，可在fit中设置
            num_stack_levels = col21.slider("num_stack_levels", min_value=0, max_value=3, value=config['train']['num_stack_levels'], step=1)
    else:
        num_bag_folds = 0
        num_stack_levels = 0
    dynamic_stacking = container.toggle("Dynamic Stacking", value = config['train']['dynamic_stacking'],
                                        help = ''' If True and num_stack_levels > 0, AutoGluon will dynamically determine whether to use stacking or not by first validating AutoGluon’s 
                                        stacking behavior. This is done to avoid so-called stacked overfitting that can make traditional multi-layer stacking, as used in AutoGluon, 
                                        fail drastically and produce unreliable validation scores. It is recommended to keep this value set to True or “auto” when using stacking, as long as 
                                        it is unknown whether the data is affected by stacked overfitting. If it is known that the data is unaffected by stacked overfitting, 
                                        then setting this value to False is expected to maximize predictive quality. If enabled, by default, AutoGluon performs dynamic stacking by 
                                        spending 25% of the provided time limit for detection and all remaining time for fitting AutoGluon. This can be adjusted by specifying ds_args 
                                        with different parameters to fit(). If “auto”, will be set to not use_bag_holdout. See the documentation of ds_args for more information.''')


    if dynamic_stacking:
        with container.expander(":rainbow[☰ Dynamic Stacking Args ☰]", expanded=True):
            # n_fold为交叉验证折数,n_repeats为交叉验证重复次数
            # n折交叉验证,设置auto_stack为True和dynamic_stacking为True
            validation_procedure = st.radio(
                "Validation procedure",
                ["holdout", "cv"],
                key=config['train']['validation_procedure'],
                horizontal=True,
            )
            if validation_procedure == "cv":
                col30,col31 = st.columns(2)
                n_folds = col30.slider("n_folds", min_value=2, max_value=10,value=config['train']['n_folds'], step=1,help='The number of folds to use for cross-validation.')
                n_repeats = col31.slider("n_repeats", min_value=1, max_value=3,value=config['train']['n_repeats'], step=1,help='The number of times to repeat the cross-validation procedure.')
                ds_args = {
                    'validation_procedure': validation_procedure,
                    'n_folds': n_folds,
                    'n_repeats': n_repeats,
                }
            else:
                ds_args = {
                    'validation_procedure': validation_procedure,
                }
    else:
        ds_args = {}


is_distill = container.checkbox(":rainbow[Distill]", value = config['train']['is_distill'], help = 'False: Ensemble learning, True: Knowledge distillation.')

if is_distill:
    col40,col41,col42 = container.columns(3)
    augmentation_data = col40.selectbox(
        "augmentation_data:", 
        selected_dataset,
        index = selected_dataset.index('Unlabeled_data'),
        placeholder = "Select augmentation_data... An optional extra dataset of unlabeled rows that can be used for augmenting the dataset used to fit student models during distillation (ignored if None)...."
    )
    teacher_preds_list = [None,'soft','hard']
    teacher_preds = col41.selectbox(
        'teacher_preds',
        teacher_preds_list,
        index = teacher_preds_list.index(config['train']['teacher_preds']),
        placeholder="Select teacher_preds... [None: only train with original labels (no data augmentation). hard: labels are hard teacher predictions given by teacher.predict(). soft: labels are soft teacher predictions given by teacher.predict_proba()]",

    )
    augment_method_list = [None,'munge','spunge']
    augment_method = col42.selectbox(
        'augment_method',
        augment_method_list,
        index=augment_method_list.index(config['train']['augment_method']),
        placeholder="Select augment_method... [None : no data augmentation performed. munge: The MUNGE algorithm. spunge: A simpler, more efficient variant of the MUNGE algorithm.]",
    )

is_train = container.button("Train")

if is_train:
    modality = 'both' if len(config['dataload']['modality']) == 2 else config['dataload']['modality'][0].lower()
    cell_subset = 'all' if len(config['dataload']['cell_subset']) == 2 else config['dataload']['cell_subset'][0].lower()
    taxonomy = config['dataload']['taxonomy'].lower()

    rowfiter='r-' + str(config['preprocess']['min_cells_per_type']+config['preprocess']['threshold_null'])
    imputer = 'i-' + config['preprocess']['imputermethod']
    scaler = 's-' + config['preprocess']['scalermethod'] if config['preprocess']['scalermethod'] else 'none'
    oversampler = 'o-' + config['preprocess']['oversamplermethod']
    test_split_rate = 't-' + str(config['preprocess']['test_split_rate'])
    
    train_method= 'distill' if is_distill else 'ensemble'

    save_path = Path('_'.join([train_method, presets, modality, taxonomy, cell_subset, rowfiter, imputer, scaler, oversampler, test_split_rate]))
    predictor = TabularPredictor(label=label_col, verbosity=verbosity,eval_metric=eval_metric, path=Path(config['default']['model_path']) / save_path, log_to_file=True,log_file_path='auto')

    if presets == 'Customize':
        with st.expander("Training Log", expanded=True),st.spinner('Wait for training...'):
            with st_stdout("code"), st_stderr("code"):
                predictor.fit(dataset[curdataset], time_limit=time_limit, excluded_model_types=excluded_model_types, auto_stack=auto_stack, num_bag_folds=num_bag_folds, 
                              num_stack_levels=num_stack_levels, dynamic_stacking=dynamic_stacking, ds_args=ds_args)
        if is_distill:
            with st.expander("Distilling Log", expanded=True),st.spinner('Wait for Distilling...'):
                with st_stdout("code"), st_stderr("code"):
                    student_models = predictor.distill(augmentation_data = dataset[augmentation_data] if augmentation_data else None, teacher_preds = teacher_preds, augment_method = augment_method, verbosity = verbosity) 
        st.balloons()
    else:
        with st.expander("Training Log", expanded=True),st.spinner('Wait for training...'):
            with st_stdout("code"), st_stderr("code"):
                predictor.fit(dataset[curdataset], presets=presets, time_limit=time_limit, excluded_model_types=excluded_model_types)
        if is_distill:
            with st.expander("Distilling Log", expanded=True),st.spinner('Wait for Distilling...'):
                with st_stdout("code"), st_stderr("code"):
                    student_models = predictor.distill(augmentation_data = dataset[augmentation_data] if augmentation_data else None, teacher_preds = teacher_preds, augment_method = augment_method, verbosity = verbosity) 
        st.balloons()
    config['train']['excluded_model_types'] = excluded_model_types
    config['train']['verbosity'] = verbosity
    config['train']['time_limit'] = time_limit
    config['train']['presets'] = presets
    config['train']['eval_metric'] = eval_metric
    if presets == 'Customize':
        config['train']['auto_stack'] = auto_stack
        if auto_stack:
            config['train']['num_bag_folds'] = num_bag_folds
            config['train']['num_stack_levels'] = num_stack_levels
        config['train']['dynamic_stacking'] = dynamic_stacking
        if dynamic_stacking:
            config['train']['validation_procedure'] = validation_procedure
            if validation_procedure == "cv":
                config['train']['n_folds'] = n_folds
                config['train']['n_repeats'] = n_repeats
    
    config['train']['is_distill'] = is_distill
    if is_distill:
        # config['train']['augmentation_data'] = augmentation_data
        config['train']['teacher_preds'] = teacher_preds
        config['train']['augment_method'] = augment_method
    toml.dump(config, open(Path(config['default']['model_path']) / save_path / 'model-config.toml', 'w'))

    if predictor.is_fit:
        viewpredictor(predictor)
    dataset[curdataset].to_csv(Path(config['default']['model_path']) / save_path / 'train_data.csv')
    curdataset = 'test_data'
    dataset[curdataset].to_csv(Path(config['default']['model_path']) / save_path / 'test_data.csv')
    st.session_state['save_path'] = save_path
    st.session_state['predictor'] = predictor

st.session_state['dataset'] = dataset
st.session_state['config'] = config
if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = 'Meta_data'
else:
    st.session_state['curdataset'] = curdataset
