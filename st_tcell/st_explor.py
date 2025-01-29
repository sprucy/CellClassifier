import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


dataset = st.session_state['dataset']
curdataset = st.session_state['curdataset']

selected_dataset = tuple(dataset.keys())
container = st.container(border=True)

curdataset = container.selectbox(
    "Current Dataset:", 
    selected_dataset,
    index=selected_dataset.index(curdataset),
    placeholder="Select Dataset...",
)
label_col_list = dataset[curdataset].columns
label_col_list.insert(0,None)
label_col = container.selectbox(
    "Label Column", 
    dataset[curdataset].columns,
    index=len(dataset[curdataset].columns)-1,
    placeholder="Select label column ...",
)
_,col1,col2,_ = container.columns([1,2,2,1])
col1.markdown(f"<h5 style='display: block; text-align: center;'><b>Dataset Row</b></h5>",unsafe_allow_html=True)
col1.markdown(f"<h3 style='display: block; color: SteelBlue; text-align: center;'>{dataset[curdataset].shape[0]}</h3>",unsafe_allow_html=True)
col2.markdown(f"<h5 style='display: block; text-align: center;'>Dataset Column</h5>",unsafe_allow_html=True)
col2.markdown(f"<h3 style='display: block;  color: SteelBlue; text-align: center;'>{dataset[curdataset].shape[1]}</h3>",unsafe_allow_html=True)
#col1.metric("Dataset Row", dataset[curdataset].shape[0], "")
#col2.metric("Dataset Column", dataset[curdataset].shape[1], "")
                    #col3.metric("Humidity", "86%", "4%")

view, info,  = container.tabs(["Data View", "Dataset Info"])

with view:
    subset=dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns
    event =st.dataframe(dataset[curdataset].style.highlight_max(subset=subset,axis=0,color='Pink')
                 .highlight_min(subset=subset,axis=0,color='Aquamarine')
                 .highlight_null(color='Linen'),
                 selection_mode=["multi-row", "single-column"], on_select="rerun")
    #print(event.selection)


    if event.selection.columns:
        if event.selection.rows:
            st.dataframe(dataset[curdataset].iloc[event.selection.rows].loc[:,event.selection.columns].describe().T, use_container_width=True)
        else:
            st.dataframe(dataset[curdataset].loc[:,event.selection.columns].describe().T, use_container_width=True)
           

with info:  
    obj_col=dataset[curdataset].select_dtypes(include=['object', 'category', 'bool']).columns
    num_col=dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns

    if obj_col.size:
        obj_col_desc=dataset[curdataset].loc[:,dataset[curdataset].select_dtypes(include=['object', 'category', 'bool']).columns].describe().T
    if num_col.size:
        num_col_desc=dataset[curdataset].loc[:,dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns].describe().T
        num_col_median=dataset[curdataset].loc[:,dataset[curdataset].select_dtypes(exclude=['object', 'category', 'bool']).columns].median()
    
    if obj_col.size and num_col.size:
        col_desc=pd.concat([obj_col_desc,num_col_desc],axis=0)
    else:
        if obj_col.size:
            col_desc=obj_col_desc
        else:
            col_desc=num_col_desc
    dtypes_col = dataset[curdataset].dtypes
    dtypes_col.name='dtypes'
    df=pd.concat([dtypes_col,col_desc],axis=1)
    event = st.dataframe(df,selection_mode=["single-row"], on_select="rerun", use_container_width=True)
    if event.selection.rows:
        col_name = df.iloc[event.selection.rows[0]].name
        if not df.iloc[event.selection.rows[0]]['dtypes'] in ['object', 'category', 'bool']:
            col1,col2 = container.columns(2)
            ax1=dataset[curdataset][col_name].plot(kind='box',title='Box Plot')
            col1.pyplot(ax1.figure,clear_figure=True)
            ax2=dataset[curdataset][col_name].plot(kind='hist',title='Histogram')
            col2.pyplot(ax2.figure,clear_figure=True)




st.session_state['dataset'] = dataset
if 'curdataset' not in st.session_state:
    st.session_state['curdataset'] = 'Meta_data'
else:
    st.session_state['curdataset'] = curdataset
