from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
from transformers import pipeline
from PIL import Image
import torchvision.models as models
import torch
import io
import torch.nn as nn

app = FastAPI()

#загрузка модели
model = models.resnet18(weights=None)              # создаем архитектуру
model.fc = nn.Linear(512, 10) # заменяем голову
model.load_state_dict(torch.load(                  # загружаем веса
    "resnet_model_cifar.pt",
    map_location='cpu'                             # у docker нет gpu
))
model.eval()                                       # переключаем в режим интерфейса


# классы ImageNet (топ-10 для примера, в реале их 1000)
# для CIFAR-10 замени на свои 10 классов
CLASSES = ["cat", "dog", "car", "airplane", "ship",
           "truck", "bird", "deer", "frog", "horse"]

# трансформация
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_index: int

class InfoResponse(BaseModel):
    model_name: str
    version: str
    author: str

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/info", response_model=InfoResponse)
def info():
    return {
        "model_name": "resnet18-imagenet",
        "version": "1.0.0",
        "author": "gregortop"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # читаем файл → PIL Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transforms(image).unsqueeze(0) # [1, 3, 224, 224]

    #предсказание
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)       #нормализуем classes, делаем из вероятностей сумму 1
        predicted_idx = probs.argmax(dim=1).item()  #достаем индекс победителя из тензора - число
        confidence = probs[0][predicted_idx].item() #из матрицы вероятностей достаем конкретное число - уверенность модели в своем выборе

    return {
        "predicted_class": CLASSES[predicted_idx % len(CLASSES)],
        "confidence": round(confidence, 4),
        "class_index": predicted_idx
    }

