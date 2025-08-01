"""
Simple startup script for Opik Integration API only
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for Opik modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import uvicorn
    from evaluation.api_integration import app
    
    print("🚀 Starting Opik Integration API Server...")
    print("📊 Dashboard will be available at: http://localhost:8000/dashboard")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🔧 Admin panel will be available at: http://localhost:8000/admin")
    print("🔗 Health check at: http://localhost:8000/health")
    print("")
    print("Press Ctrl+C to stop the server")
    print("")
    
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   py -3.10 -m pip install opik fastapi uvicorn sqlalchemy lime shap scikit-learn numpy pandas asyncio-throttle")
    input("Press Enter to exit...")
except Exception as e:
    print(f"❌ Server error: {e}")
    input("Press Enter to exit...")
