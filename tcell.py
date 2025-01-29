from pathlib import Path
import argparse
import datetime
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *

warnings.filterwarnings("ignore")

test_size = 0.2
min_cells_per_type = MIN_CELLS_PER_TYPE 
threshold = 0.2


def main(args):
    # Load data
    print('Loading data  ......')
    print('------------------------------------------------------------------------------------------------------')    
    meta_file = Path(DATA_PATH) / Path("meta_data_withVU.csv")        # read the meta data file
    ephys_file = Path(DATA_PATH) / Path("ephys_data_withVU.csv")      # read the ephys data file
    morph_file = Path(DATA_PATH) / Path("morph_data_withVU.csv")      # read the morph data file
    # Read in the data files
    meta_data = pd.read_csv(meta_file, index_col=0) 
    ephys_data = pd.read_csv(ephys_file, index_col=0)
    morph_data = pd.read_csv(morph_file, index_col=0)

    print('原始数据文件:')
    print('ephys_dataset:', ephys_data.shape)
    print('morph_dataset:', morph_data.shape)

    cell_subset = 'All'                             # choose which cells:           'all', 'superficial' or 'deep'
    modality = 'Both'                               # choose which modality:        'both', 'ephys' or 'morph'
    taxonomy = 'Berg'                               # choose which taxonomy:        'Berg', 'SEA_AD' or mouse

    #设置训练精度
    presets='medium_quality'
    eval_metric = 'accuracy'
    verbosity = 2
    time_limit = 3600

    auto_stack = False
    # bagging的折数,缺省好像为8，暂未设置，可在fit中设置
    num_bag_folds = 5
    # stacking的级别，暂未设置，可在fit中设置
    num_stack_levels = 2
    num_bag_sets = 1
    # n折交叉验证,设置auto_stack为True和dynamic_stacking为True
    dynamic_stacking = False
    # n_fold为交叉验证折数,n_repeats为交叉验证重复次数
    ds_args = {
        'n_folds': 3,
        'n_repeats': 1,
    }
    # 模型的保存路径，详细的模型命名规则见：https://auto.gluon.ai/stable/api/autogluon.tabular.models.html
    save_path = Path(MODEL_PATH + presets + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_path.mkdir(parents=True, exist_ok=True)
    fig_path = save_path / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)
    teacher_preds = 'soft'
    # augment_method 可选'munge'和'spunge',缺省为spunge，可以选择测试模型的效果
    augment_method = 'munge'

    labeled_data, unlabeled_data =  load_data(meta_data, ephys_data, morph_data, taxonomy = taxonomy, modality = modality, cell_subset = cell_subset)
    
    print('1. Data load parameter:')
    print('cell_subset: ',cell_subset)
    print('modality:    ', modality)
    print('taxonomy:    ', taxonomy)
    print('2. Loaded Original Dataset:')
    print('labeled_data:    ', labeled_data.shape)
    print('unlabeled_data:  ', unlabeled_data.shape)    

    # Original Dataset Class Distribution
    plt.figure(figsize=(10,4))
    ax = labeled_data[LABEL_COLUMN].value_counts().plot.pie(autopct='%.2f%%')
    figtitle = 'Original Dataset Class Distribution'
    ax.set_title(figtitle, fontsize=14)
    print(f'Save Figure:{figtitle}')
    plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))
    #plt.show()

    plt.figure(figsize=(10,6))
    labeled_data[LABEL_COLUMN].value_counts().plot(kind='bar')
    # 为条形图添加数值标签
    plt.bar_label(plt.gca().containers[0])
    figtitle = 'Original Dataset Number of samples per classification'
    plt.title(figtitle, fontsize=14)
    plt.xticks(rotation=30,fontsize=8)
    print(f'Save Figure:{figtitle}')
    plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))
    #plt.show()

    # 数据预处理
    print('\nPreprocessing Dataset ......')    
    print('------------------------------------------------------------------------------------------------------')

    # 数据集过滤
    if args.filter:

        print('-------------------------------            filtering Dataset            -------------------------------')
        print('过滤前数据集信息:')
        print(labeled_data.info())
        labeled_data = filter(labeled_data,min_cells_per_type=min_cells_per_type, threshold=threshold)
        print('过滤后数据集信息:')
        print(labeled_data.info())

    # 数据缺失值补插
    if args.impute_method:

        print('-------------------------------            Imputing Dataset            -------------------------------')
        print('1. Imputing Labeled Dataset')
        # 不包含标签列进行补插
        print('补插前标记数据集信息:')
        t_type_labels = labeled_data.iloc[:, -1].values
        print(labeled_data.info())
        labeled_data = impute(labeled_data.iloc[:,0:labeled_data.columns.size-1],method=args.impute_method)
        labeled_data[LABEL_COLUMN] = t_type_labels
        print('补插后标记数据集信息:')
        print(labeled_data.info())

        print('2. Imputing Unlabeled Dataset')
        # 对unlabel_data数据集补插 
        print('补插前未标记数据集信息:')
        print(unlabeled_data.info())
        unlabeled_data = impute(unlabeled_data.iloc[:,0:unlabeled_data.columns.size-1],method=args.impute_method)
        unlabeled_data[LABEL_COLUMN] = np.nan
        print('补插后未标记数据集信息:')
        print(unlabeled_data.info())
        
    # 转换为numpy数组
    data_array = labeled_data.iloc[:,0:labeled_data.columns.size-1].values              # transform the data to an array

    # Check if data needs to be scoring
    if args.scaler_method:
        print('-------------------------------             Scaling Dataset            -------------------------------')
        data_array = scale(data_array, scalemethod = args.scaler_method)                     # apply scoring method 


    cell_ids_subset = labeled_data.index                                                # extract the cell IDs of the subset
    # Create a list of features
    feature_list = labeled_data.iloc[:,0:labeled_data.columns.size-1].keys().values
    # normalized_depths = labeled_data['L23_depth_normalized'].values                     # extract normalized depth values
    t_type_labels = labeled_data.iloc[:, -1].values                                     # extract t-type labels of the data subset
    t_types_updated = np.unique(t_type_labels)
    print('dataset rownum:',data_array.shape[0])
    print('feature num:',len(feature_list))
    print('feature list:',feature_list)
    print('cell type num:',len(t_types_updated))
    print('cell type list:',t_types_updated)

    # Split the data into a training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(data_array, t_type_labels, test_size=test_size, stratify=t_type_labels, shuffle=True, random_state=RANDOM_STATE)
    # 组合X_train数据为dataframe，未包含label列
    train_data= pd.DataFrame(data = X_train, columns = feature_list)
    train_data[LABEL_COLUMN] = y_train
    if args.oversample_method:
        # 过采样的数据集不能有空值
        # 当程序未设置插值方法,这里使用默认的knn方法,否则使用设定的插值方法
        if args.impute_method:
            train_data = impute(train_data, method = args.impute_method, random_state = RANDOM_STATE)
        else:
            train_data = impute(train_data, random_state = RANDOM_STATE)
        # 转换为numpy数组
        y_train = train_data[LABEL_COLUMN].values
        X_train = train_data.iloc[:,0:train_data.columns.size-1].values
        # 进行过采样
        X_train, y_train = oversampler(X_train, y_train, method = args.oversample_method)    

        
        train_data= pd.DataFrame(data = X_train, columns = feature_list)
        train_data[LABEL_COLUMN] = y_train
        # 绘制过采样后的训练集的每分类的数量
        plt.figure(figsize=(10,6))
        train_data[LABEL_COLUMN].value_counts().plot(kind='bar')
        # 为条形图添加数值标签
        plt.bar_label(plt.gca().containers[0])
        figtitle = 'Oversampled Dataset Number of samples per classification'
        plt.title(figtitle, fontsize=14)
        plt.xticks(rotation=30,fontsize=8)
        print(f'Save Figure:{figtitle}')
        plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))
        #plt.show()   

        ax = train_data[LABEL_COLUMN].value_counts().plot.pie(autopct='%.2f%%')
        figtitle = 'Over-sampled Dataset Class Distribution'
        ax.set_title(figtitle)
        print(f'Save Figure:{figtitle}')
        plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))


    test_data_nolabel = pd.DataFrame(data = X_test, columns = feature_list)
    test_data = test_data_nolabel.copy()
    test_data[LABEL_COLUMN] = pd.DataFrame(y_test, columns = [LABEL_COLUMN])

    #import seaborn as sns
    #fig_file = fig_path / Path("trainsetscatterplot.png")
    #sns.scatterplot(x='FAP_avg_upstroke', y='Ratio TDL', data=train_data, hue=LABEL_COLUMN)
    #fig.get_figure().savefig(fig_file, dpi = 400)
    

    # 模型训练
    from autogluon.tabular import TabularDataset, TabularPredictor

    print("\nTraining model ......") 
    print('------------------------------------------------------------------------------------------------------')


    # 在模型目录中写入程序运行信息和数据预处理的基础信息
    with open(save_path / Path('model_info.txt'), 'w') as f:
        f.writelines(f'-------------------------------     Program Setup Info     -------------------------------')
        f.writelines(f'\nMIN_CELLS_PER_TYPE:{MIN_CELLS_PER_TYPE}')
        f.writelines(f'\ncell_subset:  {cell_subset}')
        f.writelines(f'\nmodality:     {modality}')
        f.writelines(f'\ntaxonomy:     {taxonomy}')
        f.writelines(f'\ndataset:      {labeled_data.shape}')
        f.writelines(f'\ntrain_data:   {X_train.shape}')
        f.writelines(f'\ntrain_label:  {np.unique(y_train)}')
        f.writelines(f'\ntest_data:    {X_test.shape}')
        f.writelines(f'\ntest_label:   {np.unique(y_test)}')
        f.writelines(f'\npresets:      {presets}')
        f.writelines(f'\n------------------------------- End of Program Setup Info -------------------------------')
    print(args)

    # 开始训练
    predictor = TabularPredictor(label=LABEL_COLUMN, verbosity=verbosity,eval_metric=eval_metric, path=save_path, log_to_file=True,log_file_path='auto')
    if auto_stack and dynamic_stacking:
        predictor.fit(train_data, presets=presets, time_limit=time_limit, auto_stack=auto_stack, num_bag_folds=num_bag_folds, 
                            num_stack_levels=num_stack_levels, num_bag_sets=num_bag_sets,dynamic_stacking=dynamic_stacking, ds_args=ds_args)
    elif auto_stack and not dynamic_stacking:
        predictor.fit(train_data, presets=presets, time_limit=time_limit, auto_stack=auto_stack, num_bag_folds=num_bag_folds,
                            num_stack_levels=num_stack_levels, num_bag_sets=num_bag_sets)
    elif not auto_stack and dynamic_stacking:
        predictor.fit(train_data, presets=presets, time_limit=time_limit, auto_stack=auto_stack, dynamic_stacking=dynamic_stacking, ds_args=ds_args)
    else:
        predictor.fit(train_data, presets=presets, time_limit=time_limit)

    # 加载保存的模型的方法
    # 加载以前训练结果，可用于查看以前的参数和使用以前训练的模型进行预测
    #predictor_path = "Models\Teacher\medium_quality-20240731-224033"
    #teacher_predictor = TabularPredictor.load(predictor_path)
    if args.train == 'distill':
  
        # 蒸馏student模型
        augmentation_data = unlabeled_data.iloc[:,0:-1]
        student_models = predictor.distill(augmentation_data = augmentation_data, teacher_preds = teacher_preds, augment_method = augment_method, verbosity = verbosity) 
        print(student_models)
        # 保存模型的信息
        predictor.save(save_path)
    # 显示训练结果信息
    print("AutoGluon infers problem type is: ", predictor.problem_type)
    print("AutoGluon identified the following types of features:")
    print(predictor.feature_metadata)
    print("Best Model:", predictor.model_best)
    result = predictor.fit_summary(show_plot=True)
    print(result)
    # Get leaderboard of models (optional)
    leaderboard = predictor.leaderboard(test_data, extra_metrics=['accuracy', 'balanced_accuracy','f1_macro','f1_micro', 'f1_weighted'],extra_info=False, silent=True)
    # Access predictions and evaluation results as needed
    print(leaderboard)


    if args.test:
        print("\nTesting model ......")
        print('------------------------------------------------------------------------------------------------------')
        from sklearn.metrics import classification_report
        y_pred = predictor.predict(test_data_nolabel)
        results = pd.concat([y_pred, pd.DataFrame(y_test, columns = [LABEL_COLUMN])], axis=1)
        results.columns=['predicted', 'actual']
        # 打印每个类的精确度，召回率，F1值, 由于样本量较少,会出现被0除, 计算结果为0，但会出现警告错误：
        # UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. 
        # Use `zero_division` parameter to control this behavior._warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
        # 输出预测值与实际值对比
        print("------------------------------- Print actual result and predict result -------------------------------")
        print(results)
        
        print("-------------------------------      Print classification report       -------------------------------")
        print(classification_report(results['actual'], results['predicted'], target_names=t_types_updated))

        print("\n-------------------------------      Ploting Confusion Matrix        -------------------------------")        
        figtitle = "Confusion Matrix"
        from sklearn.metrics import confusion_matrix
        cfm = confusion_matrix(results['actual'], results['predicted'])
        #imshow和matshow二者不同在于横轴一个在上方一个在下方，还有就是plt.matshow()显示图片可以连续使用，
        # 但是plt.imshow()想要显示多张图片必须每次都新建一个图plt.figure()或者使用plt.subplots()
        ax = plt.imshow(cfm, origin ='lower') 
        plt.title(figtitle,fontsize=25)
        plt.colorbar(ax.colorbar, fraction=0.025)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        # 设置刻度字体大小
        plt.xticks(np.arange(len(t_types_updated)),labels=t_types_updated, rotation=25,  ha="right",fontsize=10)
        plt.yticks(np.arange(len(t_types_updated)),labels=t_types_updated, fontsize=10)
        #plt.set_xticklabels(t_types_updated)
        for i in range(cfm.shape[0]):
            for j in range(cfm.shape[1]):
                plt.text(j, i, f'{cfm[i, j]:.0f}', ha='center', va='center', color='white')
        plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))
    

        # 输出预测概率
        print("-------------------------------        Print predict probability       -------------------------------")
        y_pred_proba = predictor.predict_proba(test_data)
        print(y_pred_proba)

        # 输出评估结果 
        print("-------------------------------          Print evaluate result         -------------------------------")
        eval_result = predictor.evaluate(test_data, display=True, detailed_report=True)
        print(eval_result)

        # 输出特征重要性
        print("-------------------------------         Print feature importance       -------------------------------")
        print(predictor.feature_importance(test_data))


        # 绘制特征重要性图 
        print("-------------------------------      Print feature importance plot     -------------------------------")
        fig, ax = plt.subplots(figsize=(10, 10))
        predictor.feature_importance(test_data).plot(kind='barh', ax=ax)
        figtitle = 'Feature Importance'
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(figtitle)
        plt.savefig(fig_path / Path(figtitle).with_suffix('.png'))
        
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="T-Cell Machine Learning Model Training Script")
    # Add command-line arguments
    parser.add_argument("-f", "--filter", default=True, help="filter the dataset, default True, Choose 'True', 'False'")
    parser.add_argument("-i", "--impute_method", default=None, help="Impute method, default None, Otherwise Choose 'knn', 'liner', 'Polynomial', 'decisiontree'")
    parser.add_argument("-s", "--scaler_method", default=None, help="Scaler method, default None, Otherwise Choose 'standard','maxabs','minmax', 'normalizer','power', quantile', 'robust'.")
    parser.add_argument("-o", "--oversample_method", default=None, help="Oversample method, default None, Otherwise  Choose 'random', 'smote', 'smoten', 'smotenc', 'borderline', 'adasyn', 'svmsmote', 'kmeans'")    
    parser.add_argument("-t", "--train", default='ensemble', help="Train the model, default 'ensemble', Choose 'ensemble', 'distill'")
    parser.add_argument("-e", "--test", default=True , help="Test the model, default True, Choose 'True', 'False'")

    args = parser.parse_args()
    main(args)