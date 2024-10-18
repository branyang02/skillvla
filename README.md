# Installation

1. Pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!
```

2. Install the requirements
```
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

3. Install SimplerEnv for evaluation


```
git submodule update --init --recursive
pip install -e submodules/SimplerEnv-OpenVLA/ManiSkill2_real2sim
pip install -e submodules/SimplerEnv-OpenVLA
```