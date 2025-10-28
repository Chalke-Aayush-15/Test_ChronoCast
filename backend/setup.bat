@echo off
REM ChronoCast Backend Setup Script for Windows

echo ==========================================
echo ChronoCast Backend Setup
echo ==========================================

REM Check Python
echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
echo [92mVirtual environment created[0m

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
echo [92mPip upgraded[0m

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
echo [92mRequirements installed[0m

REM Install ChronoCast
echo.
echo Installing ChronoCast library...
cd ..
pip install -e .
cd backend
echo [92mChronoCast installed[0m

REM Create .env file
if not exist .env (
    echo.
    echo Creating .env file...
    (
        echo # Django Configuration
        echo SECRET_KEY=django-insecure-dev-key-change-in-production
        echo DEBUG=True
        echo ALLOWED_HOSTS=localhost,127.0.0.1
        echo.
        echo # Database ^(SQLite for development^)
        echo USE_SQLITE=True
        echo.
        echo # For PostgreSQL, set USE_SQLITE=False and configure:
        echo # DB_NAME=chronocast_db
        echo # DB_USER=postgres
        echo # DB_PASSWORD=your_password
        echo # DB_HOST=localhost
        echo # DB_PORT=5432
    ) > .env
    echo [92m.env file created[0m
) else (
    echo .env file already exists, skipping...
)

REM Create directories
echo.
echo Creating directories...
if not exist media\datasets mkdir media\datasets
if not exist saved_models mkdir saved_models
if not exist logs mkdir logs
echo [92mDirectories created[0m

REM Run migrations
echo.
echo Running database migrations...
python manage.py makemigrations
python manage.py migrate
echo [92mMigrations complete[0m

REM Create superuser
echo.
echo Creating superuser...
echo Please enter superuser credentials:
python manage.py createsuperuser

REM Collect static files
echo.
echo Collecting static files...
python manage.py collectstatic --noinput
echo [92mStatic files collected[0m

echo.
echo ==========================================
echo [92mSetup Complete![0m
echo ==========================================
echo.
echo To start the development server:
echo   venv\Scripts\activate
echo   python manage.py runserver
echo.
echo API will be available at:
echo   http://localhost:8000/api/
echo.
echo Admin interface:
echo   http://localhost:8000/admin/
echo.
echo API Documentation:
echo   http://localhost:8000/swagger/
echo   http://localhost:8000/redoc/
echo.
echo To test the API:
echo   python test_api.py
echo.
echo ==========================================

pause