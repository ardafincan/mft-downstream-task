#!/bin/bash
# =============================================================================
# Cloud Deployment Script for Gradual Fine-tuning
# =============================================================================
# This script transfers files to a cloud server and starts training.
#
# Prerequisites:
# - SSH access to the cloud server configured (e.g., ~/.ssh/config)
# - Server has CUDA-capable GPU (e.g., A100 80GB)
#
# Usage:
#   ./deploy_to_cloud.sh [server] [options]
#
# Examples:
#   ./deploy_to_cloud.sh                    # Uses default: verda-a100
#   ./deploy_to_cloud.sh verda-a100
#   ./deploy_to_cloud.sh --sync-only
#   ./deploy_to_cloud.sh --install-only
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default Configuration
DEFAULT_SERVER="verda-a100"
PROJECT_NAME="mteb_tr_finetune"
REMOTE_BASE_DIR="~/projects"
REMOTE_PROJECT_DIR="${REMOTE_BASE_DIR}/${PROJECT_NAME}"

# Files to transfer
FILES_TO_TRANSFER=(
    "gradual_finetune.py"
    "mteb_tr_cli.py"
    ".env"
    "requirements_finetune.txt"
)

# Parse arguments
SERVER=""
SYNC_ONLY=false
INSTALL_ONLY=false
START_TRAINING=true
SCREEN_NAME="training"

print_usage() {
    echo "Usage: $0 [server] [options]"
    echo ""
    echo "Arguments:"
    echo "  server             SSH host name (default: verda-a100)"
    echo ""
    echo "Options:"
    echo "  --sync-only        Only sync files, don't install or train"
    echo "  --install-only     Only sync and install, don't start training"
    echo "  --screen <name>    Screen session name (default: training)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Deploy to verda-a100"
    echo "  $0 verda-a100"
    echo "  $0 verda-a100 --sync-only"
    echo "  $0 --install-only"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --sync-only)
            SYNC_ONLY=true
            START_TRAINING=false
            shift
            ;;
        --install-only)
            INSTALL_ONLY=true
            START_TRAINING=false
            shift
            ;;
        --screen)
            SCREEN_NAME="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
        *)
            SERVER="$1"
            shift
            ;;
    esac
done

# Use default server if not specified
if [ -z "$SERVER" ]; then
    SERVER="$DEFAULT_SERVER"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Cloud Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Server: ${GREEN}$SERVER${NC}"
echo -e "Remote directory: ${GREEN}$REMOTE_PROJECT_DIR${NC}"
echo ""

# Step 1: Test SSH connection
echo -e "${YELLOW}[0/4] Testing SSH connection...${NC}"
if ! ssh -o ConnectTimeout=10 "$SERVER" "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${RED}Error: Could not connect to $SERVER${NC}"
    echo "Please check your SSH configuration"
    exit 1
fi
echo -e "${GREEN}Connection verified!${NC}"

# Step 2: Create remote directory
echo -e "${YELLOW}[1/4] Creating remote directory...${NC}"
ssh "$SERVER" "mkdir -p $REMOTE_PROJECT_DIR"
echo -e "${GREEN}Done!${NC}"

# Step 3: Transfer files
echo -e "${YELLOW}[2/4] Transferring files...${NC}"
for file in "${FILES_TO_TRANSFER[@]}"; do
    if [ -f "$file" ]; then
        echo "  Copying $file..."
        scp "$file" "$SERVER:$REMOTE_PROJECT_DIR/"
    else
        echo -e "${YELLOW}  Warning: $file not found, skipping${NC}"
    fi
done

# Also copy the mteb package structure needed for evaluation
echo "  Syncing mteb package..."
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='results' \
    mteb/ "$SERVER:$REMOTE_PROJECT_DIR/mteb/" 2>/dev/null || \
    echo -e "${YELLOW}  Warning: Could not sync mteb package${NC}"

echo -e "${GREEN}Done!${NC}"

if [ "$SYNC_ONLY" = true ]; then
    echo -e "${GREEN}Sync completed!${NC}"
    exit 0
fi

# Step 4: Install dependencies
echo -e "${YELLOW}[3/4] Installing dependencies on remote server...${NC}"
ssh "$SERVER" << 'INSTALL_EOF'
cd ~/projects/mteb_tr_finetune

# Check if conda/venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing sentence-transformers and dependencies..."
pip install -r requirements_finetune.txt 2>/dev/null || \
    pip install sentence-transformers>=3.0.0 datasets>=2.19.0 python-dotenv wandb huggingface_hub

echo "Installing MTEB..."
pip install mteb

echo ""
echo "Installation complete!"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
INSTALL_EOF
echo -e "${GREEN}Done!${NC}"

if [ "$INSTALL_ONLY" = true ]; then
    echo -e "${GREEN}Installation completed!${NC}"
    exit 0
fi

# Step 5: Start training
if [ "$START_TRAINING" = true ]; then
    echo -e "${YELLOW}[4/4] Starting training in screen session...${NC}"
    
    # Create training script on remote
    ssh "$SERVER" << 'TRAIN_EOF'
cd ~/projects/mteb_tr_finetune
source venv/bin/activate

# Kill existing screen session if exists
screen -S training -X quit 2>/dev/null || true

# Create the training script
cat > run_training.sh << 'SCRIPT'
#!/bin/bash
cd ~/projects/mteb_tr_finetune
source venv/bin/activate

echo "=========================================="
echo "Starting Gradual Fine-tuning"
echo "Timestamp: $(date)"
echo "=========================================="
echo ""

# Run gradual fine-tuning with all features enabled
python gradual_finetune.py \
    --use-wandb \
    --push-to-hub \
    --batch-size 128 \
    --phases 0.1 0.25 0.5 1.0 \
    --epochs-per-phase 1 \
    --early-stopping-patience 1

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
SCRIPT
chmod +x run_training.sh

# Start screen session with training
screen -dmS training bash -c "./run_training.sh; exec bash"

echo ""
echo "Training started in screen session: training"
echo "To attach: screen -r training"
echo "To detach: Ctrl+A, D"
TRAIN_EOF
    echo -e "${GREEN}Done!${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Useful commands:"
echo -e "  Attach to training:  ${BLUE}ssh $SERVER -t 'screen -r training'${NC}"
echo -e "  Check GPU usage:     ${BLUE}ssh $SERVER 'nvidia-smi'${NC}"
echo -e "  View training logs:  ${BLUE}ssh $SERVER 'tail -f ~/projects/mteb_tr_finetune/gradual_finetune_output/training_results.json'${NC}"
echo ""
