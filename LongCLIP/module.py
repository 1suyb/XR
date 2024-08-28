import sys
import os
sys.path.append(os.path.dirname(__file__))
from model import longclip
import torch
import torch.nn.functional as F
from PIL import Image
import json

class LongCLIP :
    def __init__(self,
                 model='/app/LongCLIP/checkpoints/007-30--17_21_05_longclip.pt',
                 device=None) -> None:
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = longclip.load(model, device=self.device)
    
    def run(self,text=None,img=None,truncate=True) :
        text_features = None
        image_features = None
        if text is not None :
            text = longclip.tokenize(text,truncate=truncate).to(self.device)
            with torch.no_grad() :
                text_features = self.model.encode_text(text)
        if img is not None :
            img = torch.stack([self.preprocess(Image.open(im)).squeeze(0).to(self.device) for im in img])
            with torch.no_grad() :
                image_features = self.model.encode_image(img)
        
        return (text_features,image_features)
    
    def get_CosSim(self,text_features,image_features) :
        cos_sim = F.cosine_similarity(image_features.to(self.device), text_features.to(self.device))
        return cos_sim

    def get_Prob(self,text_features,image_features) :
        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1)
        return probs

