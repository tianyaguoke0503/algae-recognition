from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os

app = Flask(__name__)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 121)
model.load_state_dict(torch.load("algae_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return jsonify({'prediction': class_names[predicted.item()]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render 提供 PORT，默认为 5000
    app.run(host='0.0.0.0', port=port)