@echo off
setlocal

cd /d %~dp0

if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Upgrading pip tooling...
python -m pip install --upgrade pip setuptools wheel

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run app/app.py
