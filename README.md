suryALTO
========

A simple script to produce ALTO data based on [Surya OCR](https://github.com/VikParuchuri/surya)

## Install

```sh
pip install -r requirements.txt
```

## Run

You need to set environment variable based on your GPU. Unfortunately, right now, we can't set the batch from the command line interface directly:
- `RECOGNITION_BATCH_SIZE` is for the OCR part, I recommend `64` for 24 GB of GPU RAM
- `DETECTOR_BATCH_SIZE` is for the segmentation part, I recommend 16 for 24 GB of GPU RAM

Then you can run the script as:

```sh
RECOGNITION_BATCH_SIZE=64 DETECTOR_BATCH_SIZE=16 python to-alto.py aPDF.pdf_OR_multiple_images --destination output --lang la --format pdf/image
```

See the supported languages on the [Surya](https://github.com/VikParuchuri/surya/blob/3cdc3b69ff6571aa4d639d0778dffb56ab79159f/surya/languages.py) repository.
