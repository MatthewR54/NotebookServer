from io import BytesIO
import torch
import os
import requests
from PIL import Image 
from transformers import CLIPImageProcessor, CLIPModel
from models import UniversalFakeDetectv2
import torchvision.transforms as transforms

def is_image(img):
    return os.path.isfile(img) and img.lower().endswith(
        tuple([".jpg", ".jpeg", ".png", ".webp", ".bmp"])
    )

def real_or_fake_thres(probability, threshold=0.5):
    return "FAKE" if probability >= threshold else "REAL"

class CustomModel:
    """
    Wrapper class for the UniversalFakeDetect model.

    Initially designed to work for TrueMedia servers. Can be used in the future to interact
    with the model in a more flexible manner.
    """
    def __init__(self):
        model = UniversalFakeDetectv2()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load("pretrained_weights/continued_ft_from_concat_20240930.pth", map_location=device, weights_only=True)
        if "model" in ckpt.keys():
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

        model.eval()
        model.cuda()
        self.model = model
        clip_backbone = "openai/clip-vit-large-patch14"
        self.backbone = CLIPModel.from_pretrained(clip_backbone).to("cuda")
        self.processor = CLIPImageProcessor.from_pretrained(clip_backbone)

    def predict(self, input_data):
        file_path = input_data.get('file_path')
        
        try:
            img = self.is_valid_image(file_path)

            if not img:
                return {"error": f"Invalid media file: {file_path}. Please provide a valid image file."}
    
            img = self.processor(images=img, return_tensors="pt", size=(224, 224), do_rescale=True)["pixel_values"].cuda()
            with torch.no_grad():
                img_embeds = self.backbone.get_image_features(img)
                prob = self.model(img_embeds).sigmoid()
            
            result = {
                "df_probability": prob.item(),
                "prediction": real_or_fake_thres(prob.item())
            }
            return result
        except Exception as e:
            error_msg = f"An error occurred during inference: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    def is_valid_image(self, file_or_url):
        try:
            if file_or_url.startswith(('http://', 'https://')):
                response = requests.get(file_or_url)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(file_or_url)
        
            return img.convert('RGB')
        except Exception:
            return None

def main():
    model = CustomModel()
    test_input = {'file_path': "https://path.to.someimage.jpg"}
    output = model.predict(test_input)
    print(output)
    
if __name__=="__main__":
    main()
    