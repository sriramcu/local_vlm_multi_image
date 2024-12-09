from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import time
from combine_images import combine_images_vertically
import os
import argparse


TEMPERATURE = 0.8
TOP_K = 50  
MAX_NEW_TOKENS = 800 
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


def load_mistral_model_and_processor():
    model_path = "saved_models/"

    if os.path.exists(model_path):
        processor = LlavaNextProcessor.from_pretrained(model_path)

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto"
        )
    else:
        os.makedirs(model_path)
        processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config, device_map="auto"
        )
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, processor, device


def generate_with_model(prompt, image, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    model, processor, device = load_mistral_model_and_processor()     

    inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,  
        do_sample=True,  # Enable sampling (for diversity)
    )
    return processor.decode(output[0], skip_special_tokens=True) 


def multi_image_inference(images: list):
    prompt = f"[INST] <image>\n The above images are multiple completely distinct photographs one below the other. Describe each image individually. [/INST]"
    print(generate_with_model(prompt, combine_images_vertically(images)))
    

def single_image_inference(image):
    prompt = f"[INST] <image>\n Describe the above image. [/INST]"
    print(generate_with_model(prompt, image))        
    

def multi_image_file_inference(*image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    multi_image_inference(images)
    
    
def main():
    parser = argparse.ArgumentParser(description='Process multiple images by vertically combining them')
    parser.add_argument('image_paths', type=str, nargs='?', default=["input_images/llava_v1_5_radar.jpg", "input_images/downloaded_image.jpg"], help='Paths to the image files')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start_time = time.time()
    multi_image_file_inference(*args.image_paths)
    print(f"Time taken: {round(time.time()-start_time)} seconds")
    
    
if __name__ == "__main__":
    main()

