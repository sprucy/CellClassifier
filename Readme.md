# Requirements

## 1.Install

The GUI requires version 3.11. If you have another version installed, you need to uninstall it and install 3.11.   

Download python 3.11 from https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe   

Install python 3.11

## 2.Install Poerty

First, install poetry

```
pip install poetry
```

After installation, you can choose to modify the location of the virtual environment, you can place the virtual environment in the project directory:

```
poetry config virtualenvs.in-project true
```

Or modify the unified virtual environment storage directory:

```
poetry config virtualenvs.path
```

## 1.3 Installing Dependencies

```
poetry install
```

# 2. Execution

## 2.1 The Virtual Environment

First go to the project directory and then to the virtual environment

```
poetry shell
```

##  2.2 Run

```
streamlit run st_tcell_app.py
```