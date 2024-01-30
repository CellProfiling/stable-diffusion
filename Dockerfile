FROM  mosaicml/pytorch:1.11.0_cu115-python3.8-ubuntu20.04

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
    torchvision \
    opencv-python==4.9.0.80 \
    pytorch-lightning==1.4.2 \
    torchmetrics==0.6.0 \
    transformers==4.21.3 \
    mosaicml-streaming \
    mosaicml \
    wandb==0.15.4 \
    scikit-image \
    omegaconf
RUN pip3 uninstall -y torchtext