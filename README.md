# FaceSwap-DataGen
Generating Datasets for Face Swap Detection to Enhance Security Measures

## Overview
This project focuses on generating face swaps using real human faces. It comprises of four phases:

1. **Metadata Extraction:** Collect metadata from the images, including age, gender, race, and emotion. 
2. **Sampling:** Stratified sampling of the metadata to ensure a balanced representation.
3. **Face Detection and Feature Extraction:** Detect faces and extract features using the `FaceAnalysis` module from `InsightFace`, with `MTCNN` as a fallback.
4. **Face Swapping:** Perform face swaps based on the detected features using the InSwapper (`inswapper_128.onnx`) model.


## Prerequisites 
- Python 3.x
- pip (Python package installer)
- Inswapper model:
    * `inswapper_128.onnx`

## Setup Instructions 
1. Install dependencies 

    ```bash
    pip install -r requirements.txt
    ```
2. Add `inswapper_128.onnx` to `./models/`.

## Running the project
```bash
python main.py
```

