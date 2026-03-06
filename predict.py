import torch
import easyocr
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from pathlib import Path

#  DEVICE 
DEVICE = "cpu" 

#  PROJECT ROOT 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "finbert_text_sentiment_model.pt"

#  IMPORT VISUAL MODEL 
from src.MobileNetV2.visual_features import MobileNetV2FeatureExtractor

#  IMAGE TRANSFORM 
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#  LOAD MODELS 
def load_models():

    # Load FinBERT directly
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)

    class SentimentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 3)

        def forward(self, text_embed, visual_embed=None):
            return self.fc(text_embed)

    classifier = SentimentClassifier().to(DEVICE)

    classifier.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    finbert.eval()
    classifier.eval()

    visual_model = MobileNetV2FeatureExtractor().to(DEVICE)
    visual_model.eval()

    # OCR
    reader = easyocr.Reader(['en'], gpu=False)

    return tokenizer, finbert, classifier, visual_model, reader


#  PREDICTION FUNCTION 
def predict_sentiment_with_models(
    image_path,
    tokenizer,
    finbert,
    classifier,
    visual_model,
    reader
):

    with torch.no_grad():

        #  IMAGE 
        img = Image.open(image_path).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)

        visual_features = visual_model(img_tensor)

        #  OCR
        text_list = reader.readtext(str(image_path), detail=0)
        extracted_text = " ".join(text_list)

        if extracted_text.strip() == "":
            extracted_text = "no financial news text detected"

        #  TEXT 
        enc = tokenizer(
            extracted_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        outputs = finbert(**enc)
        cls_embed = outputs.last_hidden_state[:, 0, :]

        logits = classifier(cls_embed)
        pred = torch.argmax(logits, dim=1).item()

        label_map = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"          
        }

        return label_map[pred], extracted_text


#  TERMINAL TEST 
if __name__ == "__main__":

    tokenizer, finbert, classifier, visual_model, reader = load_models()

    image_path = input("Enter image path: ").strip()

    sentiment, text = predict_sentiment_with_models(
        image_path,
        tokenizer,
        finbert,
        classifier,
        visual_model,
        reader
    )

    print("\n📰 Extracted Text:", text)
    print("\n📊 Predicted Market Sentiment:", sentiment)
