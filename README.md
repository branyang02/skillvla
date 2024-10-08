# Installation

1. Create Conda Environment

```bash
conda create -n skillvla-env python=3.10 -y && conda activate skillvla-env
```

2. Install the requirements

```bash
pip install -r requirements.txt
```

3. Install additional requirements

```bash
pip install -e submodules/octo
```

4. Install skillvla

```bash
pip install -e .
```