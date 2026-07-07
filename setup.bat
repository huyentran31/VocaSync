@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================
::  VocaSync - setup.bat
::  Double-click to run. First time: installs everything.
::  After that: just launches the Streamlit app.
:: ============================================================

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "INSTALLED_FILE=%PROJECT_DIR%\.installed"
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"

:: ---- Already set up? -> go straight to launch ----
if exist "%INSTALLED_FILE%" if exist "%VENV_PY%" goto LAUNCH

:: ============================================================
::  FIRST-TIME SETUP
:: ============================================================
echo.
echo ============================================================
echo  FIRST-TIME SETUP - VocaSync
echo  Folder: %PROJECT_DIR%
echo ============================================================
echo.

:: ---- Check Python ----
echo [1/5] Checking Python ...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Install Python 3.11-3.13 from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during install.
    pause
    exit /b 1
)
python --version

:: ---- Create venv ----
echo [2/5] Creating virtual environment (.venv) ...
if not exist "%VENV_PY%" (
    python -m venv "%PROJECT_DIR%\.venv"
    if errorlevel 1 (
        echo ERROR: Failed to create .venv
        pause
        exit /b 1
    )
)

:: ---- Install dependencies ----
echo [3/5] Installing dependencies (a few minutes the first time) ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency install failed. Check your internet connection.
    pause
    exit /b 1
)

:: ---- Download WordNet + stopwords data (deterministic backbone; stopwords -> S15/S16
:: timestamp anchor + grounding gate; a hardcoded fallback exists but the corpus is better) ----
echo [4/5] Downloading WordNet + stopwords data (one time) ...
"%VENV_PY%" -m nltk.downloader wordnet omw-1.4 stopwords
if errorlevel 1 (
    echo WARNING: NLTK download failed. The app may not run lookups until this succeeds.
)

:: ---- Ensure .env exists (API key lives here, NEVER committed) ----
echo [5/5] Checking API key file (.env) ...
if not exist "%PROJECT_DIR%\.env" (
    copy "%PROJECT_DIR%\.env.example" "%PROJECT_DIR%\.env" >nul
    echo.
    echo  IMPORTANT: a new .env was created from the template.
    echo  Open it and paste your Gemini key after  GEMINI_API_KEY=
    echo  File: %PROJECT_DIR%\.env
    echo  Get a free key at: https://aistudio.google.com/apikey
    echo.
    echo  (The app still launches without a key: the deterministic
    echo   "Expand" action works; Ask/Mine/Explain need the key.)
    echo.
)

:: ---- Mark installed ----
echo INSTALLED=1> "%INSTALLED_FILE%"

echo.
echo ============================================================
echo  Setup complete. Launching the app ...
echo ============================================================

:LAUNCH
echo.
echo ============================================================
echo  Starting VocaSync
echo ============================================================

:: ---- Stop any OLD app still holding port 8501 ----
:: Without this, a previously-launched app keeps running and the browser reconnects to the
:: STALE server (you edit code, relaunch, but still see the old version). Kill it first so
:: this launch always serves the CURRENT code. (netstat finds the PID on :8501, taskkill ends it.)
echo  Stopping any old app on port 8501 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr LISTENING') do (
    echo    - closing stale server (PID %%p)
    taskkill /F /PID %%p >nul 2>&1
)
:: give the OS a moment to release the port
timeout /t 2 /nobreak >nul

echo  Opening http://localhost:8501 in your browser shortly...
echo  Press Ctrl+C here to stop the app.
echo.
cd /d "%PROJECT_DIR%"
:: Auto-open the browser ONLY once the server is actually listening on :8501.
:: (Old code opened after a fixed 6s; the first cold start imports heavy libs and can take
::  longer, so the tab opened before the port was up -> "localhost refused to connect" until a
::  manual refresh. Poll the TCP port every 0.5s for up to 60s, then open the browser.)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){ try{ $c=New-Object Net.Sockets.TcpClient; $c.Connect('localhost',8501); $c.Close(); Start-Process 'http://localhost:8501'; break }catch{ Start-Sleep -Milliseconds 500 } }"
"%PROJECT_DIR%\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true
pause
exit /b 0
