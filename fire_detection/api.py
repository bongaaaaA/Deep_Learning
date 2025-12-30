from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
from fire_classifier import fire_Classifier_CNN

app = FastAPI("fire cnn model for image")

# =============================== Device ============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = fire_Classifier_CNN(input_shape=3, hidden_shape=10, output_shape=2)
model.load_state_dict(torch.load("fire_cnn_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
# ====================    app   ====================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    label = "Fire" if pred == 0 else "No Fire"
    return JSONResponse({"prediction": label})

