# Datapipeline
git clone TBD
<br>
cd datapipeline
<br>
python -m venv venv
<br>
.\venv\Scripts\activate
<br>
E:\NBAPredictor\venv\Scripts\python.exe -m pip install -r requirements.txt
<br>
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\core\raw_csv_to_parquet.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\core\raw_to_joined.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\core\joined_to_rolling.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\odds\raw_html_to_csv_parquet.py

# Updating requirements.lock
E:\NBAPredictor\venv\Scripts\python.exe -m pip freeze > requirements.lock


# ML
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\ml\fnn.py
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\ml\xgb.py

# Common Errors
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "E:\NBAPredictor\venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
<br>
Solution: 
E:\NBAPredictor\venv\Scripts\python.exe -m pip uninstall torch torchvision torchaudio -y
E:\NBAPredictor\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Make pandas import AFTER torch