#!/usr/bin/env python3
"""
Opik Configuration Script

This script configures Opik for the enterprise AI chatbot.
It can work both with and without an API key.
"""

import os
import sys

def configure_opik_environment():
    """Configure Opik environment variables."""
    print("🔧 Configuring Opik Environment")
    print("=" * 50)
    
    # Set the API key directly for production
    os.environ['OPIK_API_KEY'] = 'VxSuCXVv5CRCfLu8icYJki3iM'
    
    # Set other Opik configurations
    os.environ['OPIK_PROJECT_NAME'] = 'transparent-ai-chatbot'
    os.environ['OPIK_WORKSPACE'] = 'enterprise-ai'
    
    # Test Opik configuration
    try:
        import opik
        print("✅ Opik library imported successfully")
        
        # Try to initialize client
        try:
            client = opik.Opik()
            print("✅ Opik client initialized")
        except Exception as e:
            print(f"⚠️ Opik client initialization: {e}")
            print("🔄 Will use fallback evaluation mode")
    except ImportError:
        print("❌ Opik library not found")
        return False
    
    print("\n📊 Opik Configuration Summary:")
    print(f"   API Key: {'Set' if os.getenv('OPIK_API_KEY') else 'Not Set'}")
    print(f"   Project: {os.getenv('OPIK_PROJECT_NAME')}")
    print(f"   Workspace: {os.getenv('OPIK_WORKSPACE')}")
    
    return True

if __name__ == "__main__":
    success = configure_opik_environment()
    if success:
        print("\n🚀 Opik configuration completed!")
    else:
        print("\n❌ Opik configuration failed!")
        sys.exit(1)
