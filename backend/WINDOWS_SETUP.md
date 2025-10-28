# Windows Setup Guide for ChronoCast Backend

## 🪟 Option 1: Automated Setup (Recommended)

### Using Batch File (.bat)
```powershell
# Run in PowerShell or CMD
.\setup.bat
```

### Using PowerShell Script (.ps1)
```powershell
# Allow script execution (run PowerShell as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup
.\setup.ps1
```

---

## 🔧 Option 2: Manual Setup (Step by Step)

### Step 1: Check Python Installation
```powershell
python --version
# Should show Python 3.8 or higher
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Or CMD
venv\Scripts\activate.bat
```

### Step 4: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 6: Install ChronoCast Library
```powershell
# Go to parent directory
cd ..

# Install in editable mode
pip install -e .

# Return to backend
cd backend
```

### Step 7: Create .env File
Create a file named `.env` in the backend directory:

```env
# Django Configuration
SECRET_KEY=django-insecure-dev-key-change-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (SQLite for development)
USE_SQLITE=True
```

### Step 8: Create Directories
```powershell
# Create required directories
New-Item -ItemType Directory -Force -Path "media\datasets"
New-Item -ItemType Directory -Force -Path "saved_models"
New-Item -ItemType Directory -Force -Path "logs"
```

### Step 9: Django Setup
```powershell
# Create Django project (if not exists)
django-admin startproject chronocast_api .

# Create app (if not exists)
python manage.py startapp forecast

# Run migrations
python manage.py makemigrations
python manage.py migrate
```

### Step 10: Create Superuser
```powershell
python manage.py createsuperuser
# Follow prompts to create admin user
```

### Step 11: Collect Static Files
```powershell
python manage.py collectstatic --noinput
```

### Step 12: Start Server
```powershell
python manage.py runserver
```

Server will start at: `http://localhost:8000`

---

## 🧪 Testing the API

### Option 1: Run Test Script
```powershell
# Make sure server is running in another terminal
python test_api.py
```

### Option 2: Manual Testing with PowerShell

```powershell
# Test health check
Invoke-WebRequest -Uri "http://localhost:8000/health/" | Select-Object -Expand Content

# View API endpoints
Start-Process "http://localhost:8000/swagger/"
```

---

## 🔍 Accessing the Application

- **API Base URL:** `http://localhost:8000/api/`
- **Admin Panel:** `http://localhost:8000/admin/`
- **Swagger Docs:** `http://localhost:8000/swagger/`
- **ReDoc:** `http://localhost:8000/redoc/`

---

## ⚠️ Common Windows Issues

### Issue 1: Execution Policy Error
```
.\setup.ps1 : File cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 2: Python Not Found
```
'python' is not recognized as an internal or external command
```

**Solution:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Make sure to check "Add Python to PATH" during installation
3. Restart PowerShell/CMD

### Issue 3: pip Install Fails
```
ERROR: Could not install packages due to an OSError
```

**Solution:**
```powershell
# Run PowerShell as Administrator
# Or use --user flag
pip install --user -r requirements.txt
```

### Issue 4: Port Already in Use
```
Error: That port is already in use.
```

**Solution:**
```powershell
# Use a different port
python manage.py runserver 8001

# Or find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### Issue 5: Module Import Errors
```
ModuleNotFoundError: No module named 'chronocast'
```

**Solution:**
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Install ChronoCast from parent directory
cd ..
pip install -e .
cd backend
```

### Issue 6: Database Locked (SQLite)
```
OperationalError: database is locked
```

**Solution:**
```powershell
# Stop the server
# Delete db.sqlite3
Remove-Item db.sqlite3

# Run migrations again
python manage.py migrate
```

---

## 📝 Quick Command Reference

### Virtual Environment
```powershell
# Activate
.\venv\Scripts\Activate.ps1

# Deactivate
deactivate
```

### Django Commands
```powershell
# Run server
python manage.py runserver

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Django shell
python manage.py shell
```

### Package Management
```powershell
# Install package
pip install <package-name>

# List installed packages
pip list

# Freeze requirements
pip freeze > requirements.txt
```

---

## 🚀 Starting Development

Once setup is complete:

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Start Django server
python manage.py runserver

# 3. In another terminal, run tests
python test_api.py

# 4. Open browser
Start-Process "http://localhost:8000/swagger/"
```

---

## 💡 Tips for Windows Users

1. **Use PowerShell** instead of CMD for better experience
2. **Windows Terminal** is recommended (available from Microsoft Store)
3. **VSCode** works great with Python on Windows
4. **Git Bash** can be used as alternative to PowerShell
5. For **PostgreSQL**, use the Windows installer from postgresql.org

---

## 🎯 Next Steps

After backend is running:
1. ✅ Test all API endpoints
2. ✅ Explore Admin interface
3. ✅ Check API documentation
4. 🚀 Move to React frontend (Day 17-18)

---

**Need Help?** Check the main SETUP.md or create an issue on GitHub!