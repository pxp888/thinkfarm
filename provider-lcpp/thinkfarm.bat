@echo off
rem One-command start for the self-contained thinkfarm provider app on Windows.
rem Usage: thinkfarm.bat [--model NAME]             PyQt6 dashboard GUI (default)
rem        thinkfarm.bat --headless [--model NAME]  headless daemon (Ctrl+C to stop)
rem        thinkfarm.bat --download [--model NAME]  download required model weights from the CLI
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MODE=gui"
if "%~1"=="--headless" (
    set "MODE=headless"
    shift
)

for %%A in (%*) do (
    if "%%~A"=="--download" (
        set "MODE=headless"
    )
)

set "PKG_LIST=httpx websockets"
if "%MODE%"=="gui" (
    set "PKG_LIST=!PKG_LIST! PyQt6"
)

rem Locate Python executable
set "SYS_PY="
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "SYS_PY=py -3"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "SYS_PY=python"
    )
)

rem Check if venv exists
if exist "venv\Scripts\python.exe" (
    set "PY=.\venv\Scripts\python.exe"
    set "PYW=.\venv\Scripts\pythonw.exe"
) else (
    if "%SYS_PY%"=="" (
        echo [run] ERROR: Python 3.10+ not found in PATH or via 'py' launcher.
        echo Please install Python 3.10+ from https://www.python.org/ or the Microsoft Store.
        pause
        exit /b 1
    )
    echo [run] Creating virtual environment (one-time setup)...
    %SYS_PY% -m venv venv
    if errorlevel 1 (
        echo [run] ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    set "PY=.\venv\Scripts\python.exe"
    set "PYW=.\venv\Scripts\pythonw.exe"
    echo [run] Installing dependencies (!PKG_LIST!)...
    if exist "wheelhouse\*.whl" (
        !PY! -m pip install --quiet --no-index --find-links wheelhouse !PKG_LIST!
        if errorlevel 1 (
            echo [run] Bundled wheels did not match interpreter — trying network install...
            !PY! -m pip install --quiet !PKG_LIST!
        )
    ) else (
        !PY! -m pip install --quiet !PKG_LIST!
    )
)

rem Launch the application
if "%MODE%"=="gui" (
    rem pythonw runs without opening or keeping a Command Prompt window
    if exist "%PYW%" (
        start "" "%PYW%" gui.py %*
        exit /b 0
    ) else (
        start "" "%PY%" gui.py %*
        exit /b 0
    )
)

"%PY%" app.py %*
