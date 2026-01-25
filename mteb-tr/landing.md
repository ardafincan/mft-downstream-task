```shell
apt update && apt install -y \
  build-essential \
  curl \
  git \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libffi-dev \
  liblzma-dev \
  tk-dev \
  ca-certificates

curl https://pyenv.run | bash

cat << 'EOF' >> ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF

source ~/.bashrc

pyenv --version

pyenv install 3.12.12
pyenv global 3.12.12

python --version

pip install -U pip setuptools wheel

pip install torch torchvision torchaudio

python - <<'EOF'
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

pip install psutil

pip install flash-attn --no-build-isolation

pip install distil-trainer
pip install wandb

nohup python train.py > train.log 2>&1 &
tail -f train.log
```

---

```shell
scp train.py verda-a100:~/train.py

 python evaluate_sts_tr.py --model "alibayram/distilled-sentence-transformer-c400" "magibu/embeddingmagibu-152m" "trmteb/turkish-embedding-model-fine-tuned" "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "google/embeddinggemma-300m" "ytu-ce-cosmos/turkish-e5-large" -s train
```
