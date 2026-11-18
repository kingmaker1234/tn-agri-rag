@echo off
REM Local start, no Docker (Windows)
REM
REM   run.bat            set up once, index the starter pack, start  ->  http://localhost:8000
REM   run.bat download   also download the 569 researched sources and index them (300+ files)
REM
REM Everything stays on this computer: the FAISS index is saved in backend\data\vectorstore\
REM (change with VECTOR_STORE_DIR in backend\.env). Stop with Ctrl+C; the index stays on disk.
setlocal
set PYTHONUTF8=1
if "%PORT%"=="" set PORT=8000
if "%HOST%"=="" set HOST=127.0.0.1
cd /d "%~dp0backend"

REM ---- 1. Python 3.10-3.13 ----------------------------------------------------------
set "PYEXE="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  echo [!] Python 3.10-3.13 is needed. Install it from python.org and run again.
  exit /b 1
)

REM ---- 2. virtual env + packages, reinstalled only when requirements.txt changes ------
if not exist .venv\Scripts\python.exe (
  echo Creating the Python environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 exit /b 1
)
set "PY=.venv\Scripts\python.exe"
"%PY%" scripts\local_setup.py deps-ok
if errorlevel 1 (
  echo Installing Python packages - the first run takes about a minute...
  "%PY%" -m pip install -q --upgrade pip
  "%PY%" -m pip install -q -r requirements.txt
  if errorlevel 1 exit /b 1
  "%PY%" scripts\local_setup.py deps-mark
)

REM ---- 3. OpenAI key, stored only in backend\.env which git ignores --------------------
"%PY%" scripts\local_setup.py key

REM ---- 4. UI, built once into frontend\dist and served by the backend ---------------
if not exist ..\frontend\dist\index.html (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [!] Node.js 20+ not found, so the UI can't be built. The API still runs at http://localhost:%PORT%/docs
  ) else (
    echo Building the UI - one time...
    pushd ..\frontend
    call npm ci --no-audit --no-fund
    call npm run build
    popd
  )
)

REM ---- 5. vector DB: starter pack, or the full 300+ file corpus ----------------------
if /i "%~1"=="download" (
  echo Downloading the researched sources and indexing them - this takes a while, safe to re-run...
  "%PY%" scripts\bootstrap_corpus.py --all --discover
)
"%PY%" scripts\build_index.py --dir data\seed --if-empty
if errorlevel 1 echo [!] Starter pack indexing failed - check OPENAI_API_KEY in backend\.env

echo.
echo Open http://localhost:%PORT%    Ctrl+C stops the server; the FAISS index stays saved on disk.
"%PY%" -m uvicorn app.main:app --host %HOST% --port %PORT%
