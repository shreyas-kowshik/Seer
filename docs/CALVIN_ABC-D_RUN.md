# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1F3IE95z2THAQ_lt3DKUFdRGc86Thsnc7?usp=sharing).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **calvin_dataset_path** to the directory where you have stored the CALVIN ABC-D data.
    * **checkpoint_path** to the parent directory where your experiment checkpoints are saved.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_ckpt_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)).

* :exclamation: **networkx:**
Due to compatibility issues between the networkx library in CALVIN and Python 3.10, we provide a compatible version of networkx.zip on the [website](https://drive.google.com/file/d/1z-d1SaI0rXfBtBicw1zPSsP-wE-26oLq/view?usp=sharing). Download and unzip it, then replace the existing networkx library in the following path:

## Seer
### Pre-train
```bash
bash scripts/CALVIN_ABC_D/Seer/pretrain.sh
```
### Fine-tune
```bash
bash scripts/CALVIN_ABC_D/Seer/finetune.sh
```
### Eval
```bash
bash scripts/CALVIN_ABC_D/Seer/eval.sh
```
### Scratch
```bash
bash scripts/CALVIN_ABC_D/Seer/scratch.sh
```
