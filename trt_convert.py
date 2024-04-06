import tensorrt
from vit import ViT
import torch 
import torch.onnx
import os 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

# 加载模型
model=ViT().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# pytorch 导出onnx
dummy_input=torch.randn(1,1,28,28).to(DEVICE)
torch.onnx.export(model,dummy_input,"vit.onnx",verbose=False)

# onnx 转 tensorrt
os.system('trtexec --onnx=vit.onnx --saveEngine=vit.trt --fp32')