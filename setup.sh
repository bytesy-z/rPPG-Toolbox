#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."
    #conda remove --name rppg-toolbox --all -y || exit 1
    conda create -n rppg-toolbox python=3.8 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate rppg-toolbox || exit 1

    # Use Tsinghua mirror for PyPI
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple || exit 1

    # Install PyTorch with CUDA 12.1 from official PyTorch China mirror
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
        --extra-index-url https://download.pytorch.org/whl/cu121 || exit 1

    pip install -r requirements.txt || exit 1

    cd tools/mamba || exit 1
    python setup.py install || exit 1
}


uv_setup() {
    rm -rf .venv || exit 1
    uv venv --python 3.8 || exit 1
    source .venv/bin/activate || exit 1

    ALIYUN_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple/"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"

    # Step 1: install wheel and setuptools from mirror
    uv pip install setuptools wheel -i "$ALIYUN_MIRROR" || exit 1

    # Step 2: install torch packages from torch index only
    uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
        --index-url "$TORCH_INDEX" || exit 1

    # Step 3: install other dependencies from mirror
    uv pip install -r requirements.txt -i "$ALIYUN_MIRROR" || exit 1

    # Step 4: install mamba
    #cd tools/mamba && python setup.py install || exit 1
    #cd ../.. || exit 1

    # Step 5: PyQt5 from mirror
    uv pip install PyQt5 -i "$ALIYUN_MIRROR" || exit 1
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
