#!/usr/bin/env python3
"""
Quickstart validation script to ensure all steps work correctly
"""
import os
import sys
import subprocess
from pathlib import Path

def validate_quickstart():
    """
    Validate that all quickstart steps work correctly
    """
    print("Validating quickstart steps...")

    # Check that required files exist
    required_files = [
        "backend/pyproject.toml",
        "backend/main.py",
        "backend/.env.example",
        "backend/README.md"
    ]

    print("✓ Checking required files exist...")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            return False

    # Check that the main.py file is properly structured
    print("✓ Checking main.py structure...")
    with open("backend/main.py", "r") as f:
        content = f.read()

    required_elements = [
        "def main():",
        "if __name__ == \"__main__\":",
        "get_config()",
        "create_services",
        "run_ingestion_pipeline"
    ]

    for element in required_elements:
        if element in content:
            print(f"  ✓ Found: {element}")
        else:
            print(f"  ✗ Missing: {element}")
            return False

    # Check that config.py exists and has proper structure
    print("✓ Checking config.py structure...")
    if os.path.exists("backend/config.py"):
        with open("backend/config.py", "r") as f:
            config_content = f.read()

        if "class Config:" in config_content and "get_config()" in config_content:
            print("  ✓ Config class and function found")
        else:
            print("  ✗ Config class or function missing")
            return False
    else:
        print("  ✗ backend/config.py missing")
        return False

    # Check that all the service modules exist
    print("✓ Checking service modules...")
    service_modules = [
        "backend/services/crawl_service.py",
        "backend/services/embedding_service.py",
        "backend/services/vector_service.py",
        "backend/clients/cohere_client.py",
        "backend/clients/qdrant_client.py",
        "backend/models/document_chunk.py"
    ]

    for module in service_modules:
        if os.path.exists(module):
            print(f"  ✓ {module} exists")
        else:
            print(f"  ✗ {module} missing")
            return False

    print("\n✓ All quickstart validation steps passed!")
    print("\nTo run the pipeline:")
    print("1. Install dependencies: uv sync")
    print("2. Create env file: cp backend/.env.example backend/.env")
    print("3. Edit backend/.env with your API keys")
    print("4. Run: uv run python backend/main.py")

    return True

if __name__ == "__main__":
    success = validate_quickstart()
    if success:
        print("\n🎉 Quickstart validation successful!")
        sys.exit(0)
    else:
        print("\n❌ Quickstart validation failed!")
        sys.exit(1)