import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np 
from dataset import MNIST
import random 

# 测试数据集
dataset=MNIST()

# 锁定batch size
BATCH_SIZE=1

# 读取trt模型文件
f=open('mlp.trt','rb')
model_content=f.read()

# trt运行时
runtime=trt.Runtime(trt.Logger(trt.Logger.ERROR)) 
# trt模型(engine就是model)
engine=runtime.deserialize_cuda_engine(model_content)

# trt的输入&输出张量名称
input_name=engine.get_tensor_name(0)    # 输入张量名称
output_name=engine.get_tensor_name(1)   # 输出张量名称

# 可以开始推理了!
total=0
correct=0
while True:
    # CPU输入&输出数据准备
    img,label=dataset[random.randint(0,len(dataset)-1)]    #img: (1,28,28) -> unsqueeze(0) -> (1,1,28,28)
    output=np.zeros((BATCH_SIZE,10),dtype=np.float32)   # ouput返回CPU的内存空间

    # 创建请求上下文
    context=engine.create_execution_context() 
    
    # 给input&output分配CUDA显存
    input_addr=cuda.mem_alloc_like(np.random.rand(BATCH_SIZE,1,28,28).astype(np.float32))    # 传入input显存大小
    output_addr=cuda.mem_alloc_like(np.random.rand(BATCH_SIZE,10).astype(np.float32)) # 输出output显存大小
    context.set_tensor_address(input_name,input_addr)   # 设置input tensor的cuda地址
    context.set_tensor_address(output_name,output_addr) # 设置output tensor的cuda地址

    # CUDA异步任务提交
    stream=cuda.Stream()
    cuda.memcpy_htod_async(input_addr,img.unsqueeze(0).numpy(),stream)# 1、将要推理的输入数据，从CPU拷贝到GPU，提交一个异步任务
    context.execute_async_v3(stream.handle) # 2、向cuda提交推理异步任务
    cuda.memcpy_dtoh_async(output,output_addr,stream)   # 3、从cuda拷贝回cpu内存的异步任务

    # 等待3个异步任务全部完成
    stream.synchronize()

    # 打印预测正确率
    total+=1
    if label==output[0].argmax():
        correct+=1
    if total%10000==0:
        print(f'{total}次测试，准确率{correct/total*100:.2f}%')