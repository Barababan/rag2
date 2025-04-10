@echo off
echo Medical Knowledge Base - Book Indexer
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.11 or later.
    pause
    exit /b 1
)

REM Check if books directory exists
if not exist books (
    echo Creating books directory...
    mkdir books
    echo Please place your PDF books in the books directory and run this script again.
    pause
    exit /b 1
)

REM Check if there are any PDF files
dir /b books\*.pdf >nul 2>&1
if errorlevel 1 (
    echo No PDF files found in the books directory.
    echo Please add some PDF books and run this script again.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo Installing dependencies...
    pip install python-dotenv
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo Installing/updating dependencies...
    pip install python-dotenv
    pip install -r requirements.txt --upgrade
)

REM Run the indexer
echo Starting book indexing process...
python local_indexer.py

echo.
echo Indexing complete! You can now deploy the application using deploy.ps1
pause 