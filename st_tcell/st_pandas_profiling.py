import numpy as np
import pandas as pd
from pathlib import Path
import sweetviz as sv
import streamlit as st

from utils import st_profile_report

meta_path = Path("data\meta_data_withVU.csv")        # read the meta data file
ephys_path = Path("data\ephys_data_withVU.csv")           # read the ephys data file
morph_path = Path("data\morph_data_withVU.csv")           # read the morph data file
meta_data = pd.read_csv(meta_path, index_col=0) 
ephys_data = pd.read_csv(ephys_path, index_col=0)
morph_data = pd.read_csv(morph_path, index_col=0) 
report = sv.analyze(meta_data)
report.show_html("Sweetviz_Report.html")


#st_profile_report(report)