# PowerShell script to install Python and start the Transparent AI Chatbot

Write-Host "🚀 Setting up Transparent AI Chatbot" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python is already installed: $pythonVersion" -ForegroundColor Green
        $pythonCmd = "python"
    } else {
        throw "Python not found"
    }
} catch {
    try {
        $pythonVersion = py --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python is already installed: $pythonVersion" -ForegroundColor Green
            $pythonCmd = "py"
        } else {
            throw "Python not found"
        }
    } catch {
        Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
        Write-Host "📥 Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "   Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "After installing Python, run this script again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if pip is available
try {
    pip --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip is available" -ForegroundColor Green
        $pipCmd = "pip"
    } else {
        throw "pip not found"
    }
} catch {
    try {
        & $pythonCmd -m pip --version | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ pip is available via python -m pip" -ForegroundColor Green
            $pipCmd = "$pythonCmd -m pip"
        } else {
            throw "pip not found"
        }
    } catch {
        Write-Host "❌ pip is not available" -ForegroundColor Red
        Write-Host "📥 Installing pip..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"
        & $pythonCmd get-pip.py
        Remove-Item "get-pip.py"
        $pipCmd = "$pythonCmd -m pip"
    }
}

# Install required packages
Write-Host "📦 Installing required packages..." -ForegroundColor Yellow
$packages = @(
    "fastapi",
    "uvicorn[standard]",
    "sqlalchemy",
    "lime",
    "shap",
    "scikit-learn",
    "numpy",
    "pandas",
    "asyncio-throttle",
    "transformers",
    "torch",
    "aif360",
    "fairlearn"
)

foreach ($package in $packages) {
    Write-Host "   Installing $package..." -ForegroundColor Cyan
    try {
        Invoke-Expression "$pipCmd install $package"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ $package installed successfully" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️ $package installation may have issues" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ❌ Failed to install $package" -ForegroundColor Red
    }
}

# Create necessary directories
Write-Host "📁 Creating project directories..." -ForegroundColor Yellow
$directories = @("data", "logs", "config")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   ✅ Created $dir directory" -ForegroundColor Green
    } else {
        Write-Host "   ✅ $dir directory already exists" -ForegroundColor Green
    }
}

# Create basic config file if it doesn't exist
$configFile = "config\settings.env"
if (!(Test-Path $configFile)) {
    Write-Host "⚙️ Creating configuration file..." -ForegroundColor Yellow
    $configContent = @"
# Transparent AI Chatbot Configuration

# Database Settings
DATABASE_URL=sqlite:///./data/chatbot_memory.db

# API Settings
API_HOST=localhost
API_PORT=8000
DEBUG=True

# Bias Detection Settings
ENABLE_BIAS_DETECTION=True
BIAS_THRESHOLD=0.7

# Explainability Settings
ENABLE_LIME=True
ENABLE_SHAP=True
EXPLANATION_DETAIL=medium

# Memory Settings
MEMORY_PERSISTENCE=True
CONVERSATION_HISTORY_LIMIT=100

# Search Settings
ENABLE_SEMANTIC_SEARCH=True
SEARCH_RESULTS_LIMIT=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/chatbot.log
"@
    $configContent | Out-File -FilePath $configFile -Encoding UTF8
    Write-Host "   ✅ Created $configFile" -ForegroundColor Green
    Write-Host "   📝 You can customize settings in this file" -ForegroundColor Yellow
}

# Try to start the server
Write-Host "🚀 Starting the Transparent AI Chatbot..." -ForegroundColor Green
Write-Host "🤖 Interactive mode will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📖 API docs will be available at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    # Add src to Python path and start the chatbot
    $env:PYTHONPATH = "src;$env:PYTHONPATH"
    & $pythonCmd main.py --mode api --host localhost --port 8000
} catch {
    Write-Host "❌ Failed to start server: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running 'python main.py --mode interactive' for command line mode" -ForegroundColor Yellow
}

Write-Host "👋 Setup complete!" -ForegroundColor Green
