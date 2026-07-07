@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ============================================================
::  VocaSync - run_tests.bat
::  Double-click to run the full OFFLINE test suite.
::  These tests MOCK the AI (en.call_ai) -> they cost ZERO
::  Gemini quota. No API key or internet needed (except the
::  conceptnet test, which degrades gracefully if offline).
:: ============================================================

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo ERROR: .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

set "TESTS=schema enrich extract_vocab pipeline hitl evals part_of conceptnet policy dedup_provenance gate_export stage_for_review app_boot validate_partition commit_real grounding_transcript"

set /a PASS=0
set /a FAIL=0
set "FAILED="

echo.
echo ============================================================
echo  Running offline test suite (16 tests, 0 Gemini quota)
echo ============================================================
echo.

for %%T in (%TESTS%) do (
    <nul set /p "=  test_%%T ... "
    "%VENV_PY%" "tests\test_%%T.py" >nul 2>&1
    if errorlevel 1 (
        echo FAIL
        set /a FAIL+=1
        set "FAILED=!FAILED! %%T"
    ) else (
        echo PASS
        set /a PASS+=1
    )
)

echo.
echo ============================================================
echo  RESULT: !PASS! passed, !FAIL! failed  (of 16)
if !FAIL! GTR 0 echo  FAILED:!FAILED!
if !FAIL! GTR 0 echo  Re-run a failing one to see details, e.g.:
if !FAIL! GTR 0 echo    .venv\Scripts\python tests\test_grounding_transcript.py
echo ============================================================
echo.
pause
exit /b !FAIL!
