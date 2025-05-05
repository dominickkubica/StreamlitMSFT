@echo off
echo Starting Reading Between the Lines Application
echo =============================================

:: Create .streamlit directory if it doesn't exist
if not exist .streamlit mkdir .streamlit

:: Copy the config file if it exists in the current directory
if exist config.toml copy config.toml .streamlit\config.toml

:: Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check for required packages
echo Checking for required packages...
python -c "import streamlit" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Streamlit not found. Installing required packages...
    python -m pip install streamlit pandas numpy matplotlib yfinance nltk plotly python-docx PyPDF2
)

:: Set environment variables to help with stability
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
set STREAMLIT_SERVER_TIMEOUT=300

:: Run the application with error handling
echo Starting the application...
echo If the application doesn't start in your browser, open http://localhost:8501
echo Press Ctrl+C to stop the application

:: Run Streamlit in a way that reduces WebSocket errors
streamlit run main.py --server.maxMessageSize=200 --server.enableWebsocketCompression=false --server.enableCORS=false --server.enableXsrfProtection=false
