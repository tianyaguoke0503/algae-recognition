import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
@st.cache_resource
def load_model():
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    num_classes = 121
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("algae_model.pth"))
    model.eval()
    return model

# 预测函数
def predict_algae(image, model, class_names):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# 获取类别列表
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Streamlit 界面
st.title("藻类（浮游生物）识别系统")
st.write("上传一张图片，系统将预测其类别。")

# 上传图片
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "png"])

if uploaded_file is not None:
    # 显示上传的图片
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="上传的图片", use_container_width=True)  # 改成 use_container_width

    # 加载模型并预测
    model = load_model()
    prediction = predict_algae(image, model, class_names)

    # 显示预测结果
    st.write(f"**预测结果**: {prediction}")