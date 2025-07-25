#!/bin/bash

# Setup script for CLIF to Valeos Inpatient project
# This script creates and sets up the virtual environment

echo "Setting up virtual environment for CLIF to Valeos Inpatient project..."

# Check if virtual environment already exists
if [ -d ".valeos_inpatient" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf .valeos_inpatient
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".valeos_inpatient" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .valeos_inpatient
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .valeos_inpatient/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    echo ""
    echo "To activate the environment in the future, run:"
    echo "source .valeos_inpatient/bin/activate"
    echo ""
    echo "To verify the setup:"
    echo "python -c \"import pandas; import numpy; print('Setup verified!')\""
else
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi