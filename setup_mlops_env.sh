#!/bin/bash

# Exit on any error
set -e

echo "🔧 Updating and installing required system packages..."
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    uuid-dev \
    tk-dev \
    make \
    libcurl4-openssl-dev \
    ca-certificates

echo "📥 Installing pyenv..."
if [ -d "$HOME/.pyenv" ]; then
    echo "⚠️  Removing existing pyenv directory..."
    rm -rf ~/.pyenv
fi

curl https://pyenv.run | bash

# Setup pyenv in shell
echo "🛠 Adding pyenv to shell config..."

SHELL_RC="$HOME/.bashrc"
if [[ "$SHELL" == */zsh ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

{
    echo '# pyenv setup'
    echo 'export PYENV_ROOT="$HOME/.pyenv"'
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
    echo 'eval "$(pyenv init --path)"'
    echo 'eval "$(pyenv init -)"'
    echo 'eval "$(pyenv virtualenv-init -)"'
} >> "$SHELL_RC"

# Reload shell config for current session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "🐍 Installing Python 3.9.18 via pyenv..."
pyenv install -s 3.9.18

echo "📦 Creating virtual environment 'MLOPS'..."
pyenv virtualenv 3.9.18 MLOPS

echo "✅ Activating virtual environment..."
pyenv activate MLOPS

echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing TensorFlow Extended packages..."
pip install \
    tensorflow \
    tfx \
    tensorflow-transform \
    tensorflow-metadata \
    tfx-bsl

echo "🎉 Done!"
echo "👉 To activate the environment in future sessions, run:"
echo "    pyenv activate MLOPS"
