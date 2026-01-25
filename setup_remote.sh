#!/bin/bash
set -e

echo "1. Installing python3-venv and pip..."
apt update
apt install -y python3-venv python3-pip

echo "2. Creating virtual environment..."
python3 -m venv venv

echo "3. Activating venv and installing dependencies..."
source venv/bin/activate
pip install --upgrade pip

# Install build dependencies for flash-attn
echo "   Installing build dependencies (torch, packaging, ninja, psutil, scikit-learn)..."
pip install torch>=2.0.0 packaging ninja psutil scikit-learn

# Install other requirements
echo "   Installing other requirements..."
pip install -r requirements.txt



# Uninstall sentence-transformers if installed, to use local version
echo "   Uninstalling sentence-transformers (to use local version)..."
pip uninstall -y sentence-transformers || true

echo "4. Creating .env file template..."
if [ ! -f .env ]; then
    echo "HF_TOKEN=" > .env
    echo "WANDB_API_KEY=" >> .env
    echo "Created .env template. Please edit it with your keys."
else
    echo ".env already exists, skipping."
fi

echo "Done! Run 'source venv/bin/activate' to start."
