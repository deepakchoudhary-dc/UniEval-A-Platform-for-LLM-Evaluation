#!/usr/bin/env python3
"""
Setup script for the Transparent AI Chatbot
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_step(step_name, description=""):
    """Print a setup step"""
    print(f"\n🔧 {step_name}")
    if description:
        print(f"   {description}")


def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   {e.stderr}")
        return False


def check_python_version():
    """Check Python version"""
    print_step("Checking Python version", "Requires Python 3.8+")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} found, but Python 3.8+ is required")
        return False
    
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} found")
    return True


def setup_environment():
    """Set up the environment"""
    print_step("Setting up environment", "Creating directories and environment file")
    
    # Create necessary directories
    directories = ["logs", "data", "data/search_index"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Created directory: {directory}")
    
    # Copy environment file if it doesn't exist
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("   ✓ Created .env file from .env.example")
            print("   ⚠️  Please edit .env file with your API keys")
        else:
            print("   ❌ .env.example not found")
            return False
    else:
        print("   ✓ .env file already exists")
    
    return True


def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing dependencies", "Installing required Python packages")
    
    # Install main dependencies
    if not run_command("pip install -r requirements.txt", "Installing from requirements.txt"):
        return False
    
    # Install optional NLP models
    print_step("Installing NLP models", "Downloading spaCy and NLTK data")
    
    # Download spaCy model
    run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model", check=False)
    
    # NLTK data will be downloaded automatically when needed
    print("   ✓ NLTK data will be downloaded automatically when needed")
    
    return True


def test_installation():
    """Test the installation"""
    print_step("Testing installation", "Running basic tests")
    
    try:
        # Test imports
        from config.settings import settings
        from src.core.chatbot import TransparentChatbot
        from src.core.memory import MemoryManager
        from src.core.search import SearchEngine
        
        print("   ✓ All core modules imported successfully")
        
        # Test database creation
        from src.data.database import DatabaseManager
        db = DatabaseManager()
        print("   ✓ Database manager created successfully")
        
        # Test memory manager
        memory = MemoryManager()
        print("   ✓ Memory manager created successfully")
        
        # Test search engine
        search = SearchEngine()
        print("   ✓ Search engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def show_next_steps():
    """Show next steps to the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - Add your OpenAI API key")
    print("   - Configure other settings as needed")
    print("\n2. Run the chatbot:")
    print("   python main.py --mode interactive")
    print("\n3. Or start the API server:")
    print("   python main.py --mode api")
    print("\n4. Or run the demo:")
    print("   python main.py --mode demo")
    print("\n5. Generate a model card:")
    print("   python main.py --mode model-card")
    print("\n6. Run tests:")
    print("   python -m pytest tests/")
    print("\n📚 Documentation:")
    print("   - Read README.md for detailed usage instructions")
    print("   - Check config/settings.py for configuration options")
    print("   - Explore examples/ directory for usage examples")


def main():
    """Main setup function"""
    print("🚀 Transparent AI Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("   ⚠️  Some dependencies may have failed to install")
        print("   ⚠️  Try running: pip install -r requirements.txt manually")
    
    # Test installation
    if not test_installation():
        print("   ⚠️  Installation test failed")
        print("   ⚠️  The chatbot may still work, but some features might be limited")
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
