"""
Setup Script for Comprehensive Opik Integration

This script helps set up and configure the comprehensive Opik integration system.
"""

import asyncio
import os
import sys
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpikIntegrationSetup:
    """Setup manager for Opik integration system."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.src_dir = self.root_dir / "src"
        self.evaluation_dir = self.src_dir / "evaluation"
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = [
            'opik',
            'fastapi',
            'uvicorn',
            'sqlalchemy',
            'lime',
            'shap',
            'scikit-learn',
            'numpy',
            'pandas',
            'asyncio-throttle'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} is missing")
        
        if missing_packages:
            logger.info(f"📦 To install missing packages, run:")
            logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        return True
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.src_dir,
            self.evaluation_dir,
            self.root_dir / "data",
            self.root_dir / "logs",
            self.root_dir / "config"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def create_config_file(self):
        """Create a basic configuration file."""
        config_content = """# Opik Integration Configuration

# Opik Settings
OPIK_API_KEY=your_opik_api_key_here
OPIK_PROJECT_NAME=chatbot_evaluation
OPIK_WORKSPACE=default

# Database Settings
DATABASE_URL=sqlite:///./data/evaluations.db

# API Settings
API_HOST=localhost
API_PORT=8000
DEBUG=True

# Evaluation Settings
DEFAULT_CONFIDENCE_THRESHOLD=0.8
DEFAULT_CORRECTION_THRESHOLD=0.7
MAX_CORRECTION_ITERATIONS=3

# UI Settings
REALTIME_UPDATE_INTERVAL=1000  # milliseconds
DASHBOARD_REFRESH_RATE=30000   # milliseconds

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/opik_integration.log
"""
        
        config_file = self.root_dir / "config" / "settings.env"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"⚙️ Created config file: {config_file}")
        logger.info("📝 Please update the configuration with your actual Opik API key")
    
    def create_startup_script(self):
        """Create a startup script for the API server."""
        startup_content = '''#!/usr/bin/env python3
"""
Startup script for Opik Integration API Server
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the API
try:
    import uvicorn
    from evaluation.api_integration import app
    
    if __name__ == "__main__":
        print("🚀 Starting Opik Integration API Server...")
        print("📊 Dashboard will be available at: http://localhost:8000/dashboard")
        print("📖 API docs will be available at: http://localhost:8000/docs")
        print("🔧 Admin panel will be available at: http://localhost:8000/admin")
        
        uvicorn.run(
            "evaluation.api_integration:app",
            host="localhost",
            port=8000,
            reload=True,
            log_level="info"
        )
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install opik fastapi uvicorn sqlalchemy lime shap scikit-learn numpy pandas asyncio-throttle")
    sys.exit(1)
except Exception as e:
    print(f"❌ Startup error: {e}")
    sys.exit(1)
'''
        
        startup_file = self.root_dir / "start_server.py"
        with open(startup_file, 'w') as f:
            f.write(startup_content)
        
        # Make executable on Unix systems
        if hasattr(os, 'chmod'):
            os.chmod(startup_file, 0o755)
        
        logger.info(f"🚀 Created startup script: {startup_file}")
    
    def create_test_script(self):
        """Create a simple test script."""
        test_content = '''#!/usr/bin/env python3
"""
Quick test script for Opik integration
"""

import sys
import asyncio
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def quick_test():
    """Run a quick test of the Opik integration."""
    try:
        from evaluation.enhanced_opik_evaluator import enhanced_opik_evaluator
        
        print("🧪 Running quick Opik integration test...")
        
        # Test evaluation
        result = await enhanced_opik_evaluator.evaluate_response_realtime(
            input_text="What is artificial intelligence?",
            output_text="AI is a field of computer science focused on creating intelligent machines.",
            conversation_id="test_conversation"
        )
        
        print(f"✅ Evaluation completed!")
        print(f"   Overall score: {result.overall_score:.2f}")
        print(f"   Confidence: {result.confidence_level}")
        print(f"   Requires correction: {result.requires_correction}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to run setup first and install dependencies")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    if success:
        print("🎉 Quick test passed! Opik integration is working.")
    else:
        print("⚠️ Quick test failed. Check configuration and dependencies.")
'''
        
        test_file = self.root_dir / "quick_test.py"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        logger.info(f"🧪 Created test script: {test_file}")
    
    def create_readme(self):
        """Create a comprehensive README."""
        readme_content = """# Comprehensive Opik Integration

This project provides a complete enterprise-scale integration with Opik for LLM evaluation and improvement.

## 🌟 Features

1. **API Retrieval** - Programmatic access to Opik evaluation results
2. **Real-time UI** - Live display of evaluation metrics and feedback
3. **Self-correction** - Automatic response improvement using Opik signals
4. **Explainability** - LIME/SHAP integration with Opik data synthesis
5. **Enhanced Search** - Advanced memory and search capabilities
6. **Admin Dashboard** - Comprehensive analytics and trend visualization
7. **Dynamic Model Cards** - Live performance statistics and documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install opik fastapi uvicorn sqlalchemy lime shap scikit-learn numpy pandas asyncio-throttle
```

### 2. Configure Settings
Edit `config/settings.env` and add your Opik API key:
```
OPIK_API_KEY=your_actual_api_key_here
```

### 3. Run Quick Test
```bash
python quick_test.py
```

### 4. Start the Server
```bash
python start_server.py
```

### 5. Access the System
- **API Documentation**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:8000/admin
- **Real-time UI**: http://localhost:8000/dashboard

## 📁 Project Structure

```
├── src/evaluation/           # Core evaluation modules
│   ├── enhanced_opik_evaluator.py    # Main evaluation engine
│   ├── self_correction.py            # Self-correction system
│   ├── realtime_ui.py               # UI components
│   ├── admin_dashboard.py           # Analytics dashboard
│   ├── dynamic_model_cards.py       # Model documentation
│   ├── explainability.py            # LIME/SHAP integration
│   └── api_integration.py           # FastAPI server
├── config/                   # Configuration files
├── data/                    # Database and data storage
├── logs/                    # Log files
└── test_opik_integration.py # Comprehensive test suite
```

## 🔧 API Endpoints

### Evaluation
- `POST /api/evaluation/evaluate` - Evaluate a response
- `POST /api/evaluation/evaluate-and-correct` - Evaluate and correct

### Correction
- `POST /api/correction/request` - Request correction
- `GET /api/correction/status/{id}` - Check correction status

### Search & Analytics
- `POST /api/search/evaluations` - Search evaluations
- `GET /api/analytics/conversation/{id}` - Conversation analytics
- `GET /api/dashboard/metrics` - Dashboard metrics

### Model Cards & Explanations
- `GET /api/model-card/generate` - Generate model card
- `POST /api/explain/comprehensive` - Get explanations

### Real-time
- `WebSocket /ws/evaluation` - Real-time evaluation updates

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_opik_integration.py
```

This tests all 7 integration features and generates a detailed report.

## 📊 Monitoring

The system includes comprehensive logging and monitoring:
- Real-time evaluation metrics
- Performance trends and analytics
- Error tracking and alerts
- Usage statistics

## 🛠️ Customization

Each module is designed to be modular and extensible:
- Add custom evaluation metrics
- Implement new correction strategies
- Extend UI components
- Add custom analytics

## 📚 Documentation

For detailed documentation on each component, see the inline documentation in each module.

## 🤝 Support

For issues or questions:
1. Check the logs in `logs/opik_integration.log`
2. Run the test suite to identify specific issues
3. Review the API documentation at `/docs`

## 📈 Performance

The system is designed for production use with:
- Async/await throughout for performance
- Database indexing for fast queries
- Caching for frequently accessed data
- Error handling and graceful degradation
"""
        
        readme_file = self.root_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"📚 Created README: {readme_file}")
    
    def run_setup(self):
        """Run the complete setup process."""
        logger.info("🔧 Starting Opik Integration Setup...")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.warning("⚠️ Some dependencies are missing. Install them before proceeding.")
        
        # Create directories
        self.create_directories()
        
        # Create configuration
        self.create_config_file()
        
        # Create scripts
        self.create_startup_script()
        self.create_test_script()
        
        # Create documentation
        self.create_readme()
        
        logger.info("✅ Setup completed successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Install missing dependencies (if any)")
        logger.info("2. Update config/settings.env with your Opik API key")
        logger.info("3. Run: python quick_test.py")
        logger.info("4. Run: python start_server.py")
        logger.info("5. Open: http://localhost:8000/docs")

def main():
    """Main setup function."""
    setup = OpikIntegrationSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
