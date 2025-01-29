import sys
from pathlib import Path
import numpy as np
import pandas as pd
#import toml
import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
from contextlib import contextmanager
#import streamlit as st
#from streamlit.components.v1.components import declare_component
from autogluon.tabular import TabularDataset, TabularPredictor

# depth_threshold = 0.575            # choose depth threshold
INDEX_COLUMN = 'specimen_id'       # index column name
LABEL_COLUMN = 'label'             # label column name
FREM3_LIST = ['Exc L2-3 LINC00507 FREM3']                                    # 需区分浅层和深层FREM3细胞的列表
SUPERFICIAL_T_TYPES = ['Exc L2 LAMP5 LTK', 'Exc L2-4 LINC00507 GLP2R', 'Exc L2-3 LINC00507 FREM3 superficial']
DEEP_T_TYPE = ['Exc L2-3 LINC00507 FREM3 deep', 'Exc L3-4 RORB CARM1P1', 'Exc L3-5 RORB COL22A1']
MIN_CELLS_PER_TYPE = 5            # minimum cells necessary in the training data per group/label/cell-type (e.g. 5 or 10)
THRESHOLD = 0.2
DATA_PATH = 'data\\'
MODEL_PATH = 'models\\'
RANDOM_STATE = 42
EPHYS_COL_DICT ={   'InputR': 'input_resistance',
                    'Sag1': 'sag',
                    'VmatSag1': 'VmatSag',
                    'vmbaseM': 'vmbaseM',
                    'TauM': 'tau',
                    'Rheobase': 'FAP_rheobase',
                    'NrofAPsFrstSwp': 'FAP_num_APs',
                    'ActualTCurr': 'TS1_rheobase',
                    'NrOfAPsTrSwp': 'TS1_num_APs',
                    'ActualTCurr2': 'TS2_rheobase',
                    'NrofAPtrainSwp2': 'TS2_num_APs',
                    'AdaptIndexTS': 'TS1_adp_index',
                    'AdaptIndexTS2': 'TS2_adp_index',
                    'ThreshFrstAP': 'FAP_threshold',
                    'PeakFrstAP': 'FAP_peak',
                    'MaxUpFrstAP': 'FAP_maxupstroke',
                    'MaxDwnFrstAP': 'FAP_maxdownstroke',
                    'HalfWFrstAP': 'FAP_halfwidth',
                    'UpStrkFrstAP': 'FAP_avg_upstroke',
                    'DwnStrkFrstAP': 'FAP_avg_downstroke',
                    'UpDwnStrkRatio': 'FAP_up_down_ratio',
                    'ThreshTSAP': 'TS1_threshold',
                    'MaxUpStrkTSAP': 'TS1_maxupstroke',
                    'MaxDwnStrkTSAP': 'TS1_maxdownstroke',
                    'HalfWTSAP': 'TS1_halfwidth',
                    'UpStrokeTSAP': 'TS1_avg_upstroke',
                    'DwnStrokeTSAP': 'TS1_avg_downstroke',
                    'UpDwnStrkRatioTSAP': 'TS1_up_down_ratio',
                    'ThreshTSAP2': 'TS2_threshold',
                    'MaxUpStrkTSAP2': 'TS2_maxupstroke',
                    'MaxDwnStrkTSAP2': 'TS2_maxdownstroke',
                    'UpStrokeTSAP2': 'TS2_avg_upstroke',
                    'DwnStrokeTSAP2': 'TS2_avg_downstroke',
                    'UpDwnStrkRatioTSAP2': 'TS2_up_down_ratio',
                    'HalfWTSAP2': 'TS2_peak',
                }

def load_data(meta_data, ephys_data, morph_data, taxonomy = 'Berg', modality = 'Both', cell_subset = 'All'):
    """
    T-cell data loading function.

    参数:
    meta_file (str): 元数据文件的路径。
    ephys_file (str): 电生理数据文件的路径。
    morph_file (str): 形态学数据文件的路径。
    taxonomy (str): 选择分类法用于生成标签，可选值包括 'Berg'、'SEA_AD' 或 'Mouse'，默认为 'Berg'。
    modality (str): 选择数据集模态，可选值包括 'Ephys'、'Morph' 或 'Both'，默认为 'Both'。
    cell_subset (str): 根据细胞类型选择数据子集，可选值包括 'Superficial'、'Deep' 或 'All'，默认为 'All'。

    返回:
    labeled_data (DataFrame): 包含已标记细胞数据的 DataFrame。
    unlabeled_data (DataFrame): 包含未标记细胞数据的 DataFrame。
    """

    # 新加数据为字符串类型，所以需要修改索引列类型为str，否则会报错
    morph_data.index = morph_data.index.astype(str)
    # 修改作为索引的ID列名与ephys和morph一致
    meta_data.rename(columns={"SpecimenID":INDEX_COLUMN},inplace=True)

    # 用meta_data的L23_depth_normalized列填充morph_data的L23_depth_normalized列的空值
    update_idx = morph_data[morph_data['L23_depth_normalized'].isna()].index
    update_vaule = meta_data.loc[meta_data.index.isin(update_idx)]['L23_depth_normalized'].values
    morph_data.loc[morph_data['L23_depth_normalized'].isna(),'L23_depth_normalized']=update_vaule

    # Select subset of cell IDs and update t-type labels based on taxonomy
    if taxonomy == 'Berg':
        label_col = 'SeuratMapping'
        # 由根据L23_depth_normalized设置深层和表层类别,更改为根据Is_deep_FREM3设置深层和表层类别
        meta_data.loc[meta_data['Is_deep_FREM3'] & meta_data[label_col].isin(FREM3_LIST)  , label_col] +=' deep'  
        meta_data.loc[(~ meta_data['Is_deep_FREM3']) & meta_data[label_col].isin(FREM3_LIST)  , label_col] +=' superficial'  
    elif taxonomy == 'SEA_AD':    
        label_col = 'SEA_AD'
    elif taxonomy == 'Mouse':
        label_col = 'SeuratMapping'
    else:
        raise ValueError("Invalid taxonomy. Choose 'Berg', 'SEA_AD' or 'Mouse'.")
    
    # Data file selection based on modality
    if modality in ['Ephys', 'Morph']:
        data = ephys_data if modality == 'Ephys' else morph_data                                # select the data based on modality
        data = data.sort_index()                                                                # sort the data on cell IDs
    elif modality == 'Both':
        data = pd.merge(ephys_data,morph_data,how='inner',left_index=True, right_index=True)    # merge the two dataframes   
        data = data.sort_index()                                                                # sort the data on cell IDs
    else:
        raise ValueError("Invalid modality. Choose 'Ephys', 'Morph', or 'Both'.")
    label_data = meta_data.loc[:,label_col]
    label_data=label_data.to_frame(name=LABEL_COLUMN)
    data = pd.merge(data,label_data,how='inner',left_index=True, right_index=True)

    unlabeled_data = data[data[LABEL_COLUMN]=='UNKNOWN']
    unlabeled_data.iloc[:,-1] = np.nan

    labeled_data = data[~ (data[LABEL_COLUMN]=='UNKNOWN')]

    # Filter based on cell type
    if cell_subset == 'Superficial':
        labeled_data = labeled_data[labeled_data[LABEL_COLUMN].isin(SUPERFICIAL_T_TYPES)]    
    elif cell_subset == 'Deep':
        labeled_data = labeled_data[labeled_data[LABEL_COLUMN].isin(DEEP_T_TYPE)]  
    elif cell_subset == 'All':
        labeled_data = labeled_data
    else:
        raise ValueError("Invalid cell subset. Choose 'Superficial', 'Deep', or 'All'.")
   
    return labeled_data, unlabeled_data

def load_data_l23_depth_normalized(meta_data, ephys_data, morph_data, taxonomy = 'Berg', modality = 'Both', cell_subset = 'All'):
    """
    T-cell data loading function.

    参数:
    meta_file (str): 元数据文件的路径。
    ephys_file (str): 电生理数据文件的路径。
    morph_file (str): 形态学数据文件的路径。
    taxonomy (str): 选择分类法用于生成标签，可选值包括 'Berg'、'SEA_AD' 或 'Mouse'，默认为 'Berg'。
    modality (str): 选择数据集模态，可选值包括 'Ephys'、'Morph' 或 'Both'，默认为 'Both'。
    cell_subset (str): 根据细胞类型选择数据子集，可选值包括 'Superficial'、'Deep' 或 'All'，默认为 'All'。

    返回:
    labeled_data (DataFrame): 包含已标记细胞数据的 DataFrame。
    unlabeled_data (DataFrame): 包含未标记细胞数据的 DataFrame。
    """

    # 新加数据为字符串类型，所以需要修改索引列类型为str，否则会报错
    morph_data.index = morph_data.index.astype(str)
    # 修改作为索引的ID列名与ephys和morph一致
    meta_data.rename(columns={"SpecimenID":INDEX_COLUMN},inplace=True)

    # 用meta_data的L23_depth_normalized列填充morph_data的L23_depth_normalized列的空值
    update_idx = morph_data[morph_data['L23_depth_normalized'].isna()].index
    update_vaule = meta_data.loc[meta_data.index.isin(update_idx)]['L23_depth_normalized'].values
    morph_data.loc[morph_data['L23_depth_normalized'].isna(),'L23_depth_normalized']=update_vaule

    # Select subset of cell IDs and update t-type labels based on taxonomy
    if taxonomy == 'Berg':
        label_col = 'SeuratMapping'
        # 由根据L23_depth_normalized设置深层和表层类别,更改为根据Is_deep_FREM3设置深层和表层类别
        meta_data.loc[meta_data['Is_deep_FREM3'] & meta_data[label_col].isin(FREM3_LIST)  , label_col] +=' deep'  
        meta_data.loc[(~ meta_data['Is_deep_FREM3']) & meta_data[label_col].isin(FREM3_LIST)  , label_col] +=' superficial'  
    elif taxonomy == 'SEA_AD':    
        label_col = 'SEA_AD'
    elif taxonomy == 'Mouse':
        label_col = 'SeuratMapping'
    else:
        raise ValueError("Invalid taxonomy. Choose 'Berg', 'SEA_AD' or 'Mouse'.")
    
    # Data file selection based on modality
    if modality in ['Ephys', 'Morph']:
        data = ephys_data if modality == 'Ephys' else morph_data                                # select the data based on modality
        data = data.sort_index()                                                                # sort the data on cell IDs
    elif modality == 'Both':
        data = pd.merge(ephys_data,morph_data,how='inner',left_index=True, right_index=True)    # merge the two dataframes   
        data = data.sort_index()                                                                # sort the data on cell IDs
    else:
        raise ValueError("Invalid modality. Choose 'Ephys', 'Morph', or 'Both'.")
    # add 'L23_depth_normalized' to Ephys data
    if modality == 'Ephys':
        L23_depth_normalized = meta_data.loc[:,'L23_depth_normalized']
        L23_depth_normalized = L23_depth_normalized.to_frame(name='L23_depth_normalized')
        data = pd.merge(data,L23_depth_normalized,how='inner',left_index=True, right_index=True)

    label_data = meta_data.loc[:,label_col]
    label_data=label_data.to_frame(name=LABEL_COLUMN)
    data = pd.merge(data,label_data,how='inner',left_index=True, right_index=True)

    unlabeled_data = data[data[LABEL_COLUMN]=='UNKNOWN']
    unlabeled_data.iloc[:,-1] = np.nan

    labeled_data = data[~ (data[LABEL_COLUMN]=='UNKNOWN')]

    # Filter based on cell type
    if cell_subset == 'Superficial':
        labeled_data = labeled_data[labeled_data[LABEL_COLUMN].isin(SUPERFICIAL_T_TYPES)]    
    elif cell_subset == 'Deep':
        labeled_data = labeled_data[labeled_data[LABEL_COLUMN].isin(DEEP_T_TYPE)]  
    elif cell_subset == 'All':
        labeled_data = labeled_data
    else:
        raise ValueError("Invalid cell subset. Choose 'Superficial', 'Deep', or 'All'.")

    return labeled_data, unlabeled_data


def filter(df, min_cells_per_type = MIN_CELLS_PER_TYPE, threshold = THRESHOLD):
    # Filter based on min_cells_per_type
    df = df.groupby(LABEL_COLUMN).filter(lambda x: len(x) >= min_cells_per_type)
    # Calculate the threshold for threshold missing values
    threshold = threshold * df.shape[1]
    # Filter out rows with more than threshold missing values
    df_filtered = df[df.isnull().sum(axis=1) <= threshold]
    
    return df_filtered

def impute(df, method='knn', **kwargs):
    """
    这个函数首先根据给定的阈值过滤掉缺失值过多的行,然后使用指定的方法填充剩余行中的缺失值
    
    参数:
    df: 需要处理的 DataFrame
    method: 填充缺失值的方法,默认为 'knn',可选项:'knn', 'linear', 'polynomial', 'decisiontree', 'mice', 'autoencoder'
    kwargs: 其他需要的参数,如 n_neighbors, max_iter, random_state 等,具体取决于所使用的方法
    
    返回:
    一个处理后的 DataFrame,其中缺失值已被填充。
    """
    # Impute the missing values in the filtered dataset
    if method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    #elif method == 'mice':
    #    df_imputed = pd.DataFrame(fancyimpute.MICE().fit_transform(df), columns=df.columns)
    # elif method == 'autoencoder':# 该方法无效,待修改或删除
    #    from keras.models import Model
    #    from keras.layers import Input, Dense
    #    input_dim = df.shape[1]
    #    input_layer = Input(shape=(input_dim,))
    #    encoded = Dense(128, activation='relu')(input_layer)
    #    encoded = Dense(64, activation='relu')(encoded)
    #    encoded = Dense(32, activation='relu')(encoded)
    #    decoded = Dense(64, activation='relu')(encoded)
    #    decoded = Dense(128, activation='relu')(decoded)
    #    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    #    autoencoder = Model(input_layer, decoded)
    #    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    #    autoencoder.fit(df_filtered, df_filtered, epochs=kwargs.get('epochs', 50), batch_size=kwargs.get('batch_size', 256), shuffle=True)
    #    df_imputed = pd.DataFrame(autoencoder.predict(df_filtered), columns=df.columns)
    else:
        # 使用多变量插补,需要显式导入enable_iterative_imputer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        print('method:',method)

        if method == 'linear':
            from sklearn.linear_model import BayesianRidge
            # 创建基于线性回归的估计器
            estimator = BayesianRidge()
            
        elif method == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            from sklearn.linear_model import LinearRegression
            # 创建基于多项式回归的估计器
            estimator = PolynomialFeatures(include_bias=False)
            estimator = make_pipeline(PolynomialFeatures(degree=kwargs.get('degree', 2), interaction_only=False, include_bias=False), LinearRegression())

        elif method == 'decisiontree':
            from sklearn.tree import DecisionTreeRegressor
            # 创建基于决策树回归的估计器
            estimator = DecisionTreeRegressor(max_features='sqrt', random_state=kwargs.get('random_state', RANDOM_STATE))
        elif method == 'randomforest':
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                # We tuned the hyperparameters of the RandomForestRegressor to get a good
                # enough predictive performance for a restricted execution time.
                n_estimators=4,
                max_depth=10,
                bootstrap=True,
                max_samples=0.5,
                n_jobs=2,
                random_state=kwargs.get('random_state', RANDOM_STATE)
            ),
        print('estimator:',estimator)
        # 创建 IterativeImputer 对象,并设置参数
        imputer = IterativeImputer(estimator=estimator, max_iter=kwargs.get('max_iter', 10), random_state=kwargs.get('random_state', RANDOM_STATE))
        # 使用 imputer 对象来填充缺失数据
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    return df_imputed
def scale(X, scalemethod='standard', **kwargs):
    """
    数据标准化函数

    参数:
    X:特征矩阵(二维数组)
    method:标准化方法(默认是'standard')。可选值包括:'standard','minmax','robust','normalizer','quantile'
    **kwargs:所选方法的关键字参数

    返回:
    X_scaled:标准化后的特征矩阵(二维数组)
    """
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler
    # 选择标准化方法
    if scalemethod =='standard':
        scaler = StandardScaler()
    elif scalemethod =='maxabs':
        scaler = MaxAbsScaler()
    elif scalemethod =='minmax':
        scaler = MinMaxScaler()
    elif scalemethod =='normalizer':
        scaler = Normalizer(norm = kwargs.get('norm', 'l2'))
    elif scalemethod =='power':
        scaler = PowerTransformer(method = kwargs.get('method', 'yeo-johnson')) 
    elif scalemethod =='quantile':
        scaler = QuantileTransformer(random_state = kwargs.get('random_state', RANDOM_STATE), output_distribution = kwargs.get('output_distribution', 'uniform'))
    elif scalemethod =='robust':
        scaler = RobustScaler(quantile_range = kwargs.get('quantile_range', (25, 75)))
    else:
        raise ValueError("Invalid scaling method. Choose'standard','maxabs','minmax', 'normalizer','power', quantile', 'robust'.")
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def oversampler(X, y, method='smote', **kwargs):
    """
    不平衡数据的过采样函数
    
    参数:
    X:特征矩阵(二维数组)
    y:标签向量(一维数组)
    method:过采样方法(默认是SMOTE)。可选值包括:'random', 'smote', 'smoten', 'smotenc', 'borderline', 'adasyn', 'svmsmote', 'kmeans'
    **kwargs:所选方法的关键字参数
    
    返回:
    X_res:经过采样后的特征矩阵(二维数组)
    y_res:经过采样后的标签向量(一维数组)
    """
    from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, RandomOverSampler, ADASYN
    if method == 'random':
        sampler = RandomOverSampler(random_state = kwargs.get('random_state', RANDOM_STATE)) 
    elif method =='smote':
        sampler = SMOTE(random_state = kwargs.get('random_state', RANDOM_STATE), k_neighbors = kwargs.get('k_neighbors', 3))
    elif method =='smoten':
        sampler = SMOTEN(random_state = kwargs.get('random_state', RANDOM_STATE), k_neighbors = kwargs.get('k_neighbors', 3))
    elif method =='smotenc':
        sampler = SMOTENC(random_state = kwargs.get('random_state', RANDOM_STATE), k_neighbors = kwargs.get('k_neighbors', 3))
    elif method =='borderline':
        sampler = BorderlineSMOTE(random_state = kwargs.get('random_state', RANDOM_STATE), k_neighbors = kwargs.get('k_neighbors', 3), kind=kwargs.get('kind', 'borderline-1'))
    elif method =='adasyn':
        sampler = ADASYN(random_state = kwargs.get('random_state', RANDOM_STATE),n_neighbors = kwargs.get('n_neighbors',3))
    elif method =='svmsmote':
        sampler = SVMSMOTE(random_state = kwargs.get('random_state', RANDOM_STATE),k_neighbors = kwargs.get('k_neighbors', 3), m_neighbors = kwargs.get('m_neighbors', 5))
    elif method =='kmeans':
        from sklearn.cluster import MiniBatchKMeans
        kmeans_estimator=MiniBatchKMeans(n_clusters=1, n_init=1, random_state=0)
        sampler = KMeansSMOTE(kmeans_estimator=kmeans_estimator,random_state = kwargs.get('random_state', RANDOM_STATE), k_neighbors = kwargs.get('k_neighbors', 3))
    
    X_res, y_res = sampler.fit_resample(X, y)
    
    return X_res, y_res

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs) #matshow

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar.ax.tick_params(labelsize=4)


    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=4)


    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=4)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# ---------------------------------------------------------------------------------------------------------------
# streamlit function
'''
EDITOR_KEY_PREFIX = '__'


def data_editor_key(key):
    """Gets data_editor key based on a value_key. """
    return EDITOR_KEY_PREFIX + key


def data_editor_change(key, editor_key):
    """Callback function of data_editor. """
    st.session_state[key] = apply_de_change(st.session_state[key], st.session_state[editor_key])


def apply_de_change(df0, changes):
    """Apply changes of data_editor."""
    add_rows = changes.get('added_rows')
    edited_rows = changes.get('edited_rows')
    deleted_rows = changes.get('deleted_rows')

    for idx, row in edited_rows.items():
        for name, value in row.items():
            df0.loc[df0.index[idx], name] = value

    df0.drop(df0.index[deleted_rows], inplace=True)

    ss = []
    has_index = add_rows and '_index' in add_rows[0]
    for add_row in add_rows:
        if '_index' in add_row:
            ss.append(pd.Series(data=add_row, name=add_row.pop('_index')))
        else:
            ss.append(pd.Series(data=add_row))
    df_add = pd.DataFrame(ss)

    return pd.concat([df0, df_add], axis=0) if has_index else pd.concat([df0, df_add], axis=0, ignore_index=True)

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            try:
                buffer.write(b)
                #buffer.write(b+"\r\n")
                output_func(buffer.getvalue())
                # over output to the placeholder
                #buffer.seek(0) # returns pointer to 0 position
                #output_func(b)
            except:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield

@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


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
'''


'''    
hide=st.button("Hide")
if hide:
    hide_pages(["Prediction"])

show=st.button("Show")
if show:
    hide_pages([])
'''