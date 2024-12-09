from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests
import time
from combine_images import combine_images_vertically
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_mistral_model_and_processor():
    model_path = "saved_models/"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    if os.path.exists(model_path):
        processor = LlavaNextProcessor.from_pretrained(model_path)

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto"
        )
    else:
        os.makedirs(model_path)
        processor = LlavaNextProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config, device_map="auto"
        )
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, processor


def multi_image_inference(images: list):
    model, processor = load_mistral_model_and_processor()    
    temperature = 0.8  # You can experiment with values like 0.7, 0.8, or 1.0
    top_k = 50  # Top-k sampling: Restrict to top 50 most likely next tokens
    max_new_tokens = 800  
    prompt = f"[INST] <image>\n The above images are multiple completely distinct photographs one below the other. Describe each image individually. [/INST]"

    inputs = processor(images=combine_images_vertically(images), text=prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,  
        do_sample=True,  # Enable sampling (for diversity)
    )
    print(processor.decode(output[0], skip_special_tokens=True))    
    
    
def single_image_inference(image):
    model, processor = load_mistral_model_and_processor()

    temperature = 0.8  # You can experiment with values like 0.7, 0.8, or 1.0
    top_k = 50  # Top-k sampling: Restrict to top 50 most likely next tokens
    max_new_tokens = 800  
    prompt = f"[INST] <image>\n Describe the above image. [/INST]"

    inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,  
        do_sample=True,  # Enable sampling (for diversity)
    )
    print(processor.decode(output[0], skip_special_tokens=True))  
    
    
def multi_image_file_inference():
    image1 = Image.open("llava_v1_5_radar.jpg")
    image2 = Image.open("downloaded_image.jpg")
    multi_image_inference([image1, image2])
    
    
def main():
    start_time = time.time()
    multi_image_file_inference()
    print(f"Time taken: {round(time.time()-start_time)} seconds")
    
    
if __name__ == "__main__":
    main()

