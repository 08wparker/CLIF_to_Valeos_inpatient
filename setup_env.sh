#!/bin/bash

# Setup script for CLIF to Valeos Inpatient project
# This script creates and sets up the virtual environment
# USAGE: source ./setup_env.sh (to activate in current shell)
#    OR: ./setup_env.sh && source .valeos_inpatient/bin/activate

echo "Setting up virtual environment for CLIF to Valeos Inpatient project..."

# Check if virtual environment already exists
NEEDS_INSTALL=false
if [ -d ".valeos_inpatient" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf .valeos_inpatient
        NEEDS_INSTALL=true
    else
        echo "Using existing virtual environment."
        # Check if dependencies need to be installed
        if [ ! -f ".valeos_inpatient/.setup_complete" ]; then
            NEEDS_INSTALL=true
        fi
    fi
else
    NEEDS_INSTALL=true
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".valeos_inpatient" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .valeos_inpatient
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        return 1 2>/dev/null || exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .valeos_inpatient/bin/activate

# Install or upgrade dependencies if needed
if [ "$NEEDS_INSTALL" = true ]; then
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        # Mark setup as complete
        touch .valeos_inpatient/.setup_complete
        echo "✅ Setup completed successfully!"
    else
        echo "❌ Error: Failed to install dependencies"
        return 1 2>/dev/null || exit 1
    fi
else
    echo "✅ Dependencies already installed, skipping..."
fi

echo ""
echo "Virtual environment is now active!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "source .valeos_inpatient/bin/activate"
echo ""
echo "To verify the setup:"
echo "python -c \"import pandas; import numpy; print('Setup verified!')\""