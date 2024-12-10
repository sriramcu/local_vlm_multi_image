# Local Visual LLM

Analyse multiple images simultaneously using local inference on a multimodal LLM - `llava-hf/llava-v1.6-mistral-7b-hf`. 

This project uses only LLM inference. The inference is completely offline, without any API keys required, thereby ensuring privacy and security.

## Setup Instructions

### Ubuntu 20.04

1. Sign up for a HuggingFace account and setup an access token. Save this token for later use. 
2. Setup CUDA 12.1, CUDNN 9.3, PyTorch '2.4.1+cu121', TorchVision 0.19.1+cu121, transformers 4.47.0, use Python 3.10.
3. `pip install -r requirements.txt`
4. `pip install huggingface_hub` for logging in via the terminal.
5. Login to your HuggingFace account in your Ubuntu terminal where you are running the program, by running `huggingface-cli login` and enter your access token saved earlier. 
6. Request access to the gated repo for `llava-hf/llava-v1.6-mistral-7b-hf` on the Huggingface portal, by filling in your details.

## Usage

`python mistral_inference.py input_images/llava_v1_5_radar.jpg input_images/downloaded_image.jpg`

Provide any number of image paths as arguments. These images will be vertically combined and fed into the LLM for inference.

**On the first run**, the model will be downloaded and saved for future use in the `saved_models` directory. This will take a few minutes and requires internet connection.   

**On subsequent runs**, the model will be loaded from the saved directory. The program does this by automatically checking for the existence of the `saved_models` directory. When the model is loaded from this directory, the inference can be run offline, i.e. without any internet connection.

**Note:** The program uses Noisy Float 4-bit quantization of the model.
