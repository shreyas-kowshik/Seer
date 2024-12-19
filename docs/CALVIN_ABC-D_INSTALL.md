# Installation

**(1) Env**
```python
conda create -n seer python=3.10
conda activate seer
```

**(2) CALVIN**
> Exact instructions on [CALVIN](https://github.com/mees/calvin).
```python
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
```

**(3) Dataset Download**
> We only download CALVIN ABC-D.
```python
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

**(4) Third Party Packages**
```python
cd ${YOUR_PATH_TO_SEER}
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
