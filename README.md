# Datapipeline
git clone TBD
<br>
cd datapipeline
<br>
python -m venv venv
<br>
.\venv\Scripts\activate
<br>
E:\NBAPredictor\venv\Scripts\python.exe -m pip install -r requirements.lock
<br>
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\static_data\raw_csv_to_parquet.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\static_data\raw_to_joined.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\datapipeline\static_data\joined_to_rolling.py

E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\ml\normalize.py

# Updating requirements.lock
E:\NBAPredictor\venv\Scripts\python.exe -m pip freeze > requirements.lock


# ML
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\ml\test.py
E:\NBAPredictor\venv\Scripts\python.exe E:\NBAPredictor\ml\llm.py