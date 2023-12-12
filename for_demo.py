import streamlit as st
import pandas as pd
import torch
from torch import nn
from torch.nn import Parameter

import random

import numpy as np


# 启动
# streamlit run for_demo.py --server.port=9002
markdown_code_1 = """
# AI模型创建及训练——入门篇
## 方程求解
假设求解方程：$y = 3x + 1$中的各个系数

也就是求解方程 $y=ax+b$ 中的$a,b$
"""
st.markdown(markdown_code_1)


code_1 = """
import pandas as pd
import numpy as np
import random

import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from tqdm import tqdm

"""

code_1a = """
point_num = 50
x = np.linspace(0, 10, point_num)
# 数据集添加扰动
y = [3*i+1 + random.random()*2 for i in x]
"""
with st.expander("准备数据"):
    st.code(code_1, language='python')

    st.code(code_1a, language='python')


point_num = st.slider("样本的数量", 10, 100, 50, key="point_num_1")

noise = st.slider("噪音", 0, 100, 2, key="noise_1")

scatter_size = st.slider("点的大小", 1, 50, 20, key="scatter_size_1")

x = np.linspace(0, 10, point_num)
# 数据集添加扰动
y = [3*i + 1 + random.random()*noise for i in x]

st.scatter_chart(pd.DataFrame({"x": x, "y":y}), x="x", y="y", size=scatter_size)


code_2 = """
# 创建模型
class DemoModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 变量
        self.a = Parameter(torch.tensor([1.]))
        self.b = Parameter(torch.tensor([2.]))
    
    def forward(self, x):
        return self.a * x+ self.b
model = DemoModel()
"""



# 创建模型
class DemoModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 变量
        self.a = Parameter(torch.tensor([1.]))
        self.b = Parameter(torch.tensor([2.]))
    
    def forward(self, x):
        return self.a * x+ self.b


code_2a='''
# 将数据改为batch输出
def get_batch_data(data, batch_size, shuffle=True):
    """
    生成 batch 数据
    :param data: 数据集
    :param batch_size: batch 大小
    :param shuffle: 是否打乱
    :return:
    """
    data_size = len(data)
    num_batches = data_size // batch_size
    # 是否需要打乱数据
    if shuffle:
        random.shuffle(data)
    out = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, data_size)
        batch_data = data[start:end]
        # 拆分 x 和 y
        x_batch, y_batch = zip(*batch_data)
        out.append([list(x_batch), list(y_batch)])
    return out
'''




def get_batch_data(data, batch_size, shuffle=True):
    """
    生成 batch 数据
    :param data: 数据集
    :param batch_size: batch 大小
    :param shuffle: 是否打乱
    :return:
    """
    data_size = len(data)
    num_batches = data_size // batch_size
    # 是否需要打乱数据
    if shuffle:
        random.shuffle(data)
    out = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, data_size)
        batch_data = data[start:end]
        # 拆分 x 和 y
        x_batch, y_batch = zip(*batch_data)
        out.append([list(x_batch), list(y_batch)])
    return out


code_3 = """
optimizer = torch.optim.Adam(params=[p for n, p in model.named_parameters()], 
                             lr=0.001)

epochs = 40
for epoch in range(epochs):
    for d, t in get_batch_data(data, 20, True):
        out = model(torch.tensor([d]))
        loss = torch.mean(torch.pow(out-torch.tensor(t),2))
        # 优化器梯度清除
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化更新
        optimizer.step()
        
"""

with st.expander("创建模型并训练"):
    st.code(code_2, language='python')
    st.code(code_2a, language='python')
    st.code(code_3, language='python')

batch_size = st.slider("batch", 1, 100, 20, key="batch_size")

# 学习率
lr = st.select_slider('learning_rate',options=[5e-5, 5e-4, 0.01, 0.02, 0.1, 0.2, 0.4], value=0.01, key="lr")

# epoch
epochs = st.slider("epochs", 1, 200, 20, key="epochs")

# shuffle
shuffle = st.toggle('打乱数据集', value=True, key="shuffle")


model = DemoModel()
a_item = []
a_grad = []
loss_item = []
# 优化器
optimizer = torch.optim.Adam(params=[p for n, p in model.named_parameters()], lr=lr)


progress_text = "training... Please wait."
training_bar = st.progress(0, text=progress_text)


data = list(zip(x, y))

for epoch in range(epochs):
    for d, t in get_batch_data(data, batch_size, shuffle):
        out = model(torch.tensor([d]))

        loss = torch.mean(torch.pow(out-torch.tensor(t),2))
        loss_item.append(loss.item())
        # 优化器梯度清除
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        a_grad.append(model.a.grad.item())
        # 优化更新
        optimizer.step()
        a_item.append(model.a.item())
        
        training_bar.progress((epoch+1)/epochs, text=progress_text)
training_bar.empty()

st.line_chart(pd.DataFrame({"变量a": a_item}), y="变量a")
st.line_chart(pd.DataFrame({"变量a梯度": a_grad}), y="变量a梯度")
st.line_chart(pd.DataFrame({"损失值loss": loss_item}), y="损失值loss")



markdown_code_1 = """
## 复杂方程
那么如何拟合一个函数：$y = 3x^2 + 5$
"""

st.markdown(markdown_code_1)


code_5 = """
point_num = 50
x = np.linspace(0, 10, point_num)
# 数据集添加扰动
y = [3*i**2 + 5 + random.random()*2 for i in x]
"""
with st.expander("准备数据"):
    st.code(code_5, language='python')




point_num_2 = st.slider("样本的数量", 10, 100, 50, key="point_num_2")

noise_2 = st.slider("噪音", 0, 100, 2, key="noise_2")

scatter_size_2 = st.slider("点的大小", 1, 50, 20, key="scatter_size_2")

x_sqr = np.linspace(0, 10, 100, dtype=np.float32)
# 数据集添加扰动
y_sqr = np.array([3*i**2 + 5 + random.random()*noise_2 for i in x_sqr], dtype=np.float32)

st.scatter_chart(pd.DataFrame({"x": x_sqr, "y":y_sqr}), x="x", y="y", size=scatter_size_2)


class DNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Linear(1,4)
        self.layers = nn.Sequential()
        for _ in range(2):
            self.layers.append(nn.Linear(4,4))
            self.layers.append(nn.ReLU())
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        upper_dim_out = self.input(x)
        layer_out = self.layers(upper_dim_out)
        return self.out(layer_out)

dnn_model = DNNModel()



code_4 = """
# 创建模型
class DNNModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Linear(1,4)
        self.layers = nn.Sequential()
        for _ in range(2):
            self.layers.append(nn.Linear(4,4))
            self.layers.append(nn.ReLU())
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        upper_dim_out = self.input(x)
        layer_out = self.layers(upper_dim_out)
        return self.out(layer_out)
dnn_model = DNNModel()
print(dnn_model)

"""



code_6 = """
optimizer = torch.optim.Adam(params=[p for n, p in model.named_parameters()], 
                             lr=0.001)

epochs = 40
for epoch in range(epochs_2):
    for d, t in get_batch_data(list(zip(x_sqr, y_sqr)), 5, True):
        out = dnn_model(torch.unsqueeze(torch.tensor(d), dim=1))
        loss = torch.mean(torch.pow(out - torch.unsqueeze(torch.tensor(t),dim=1),2))
        loss_item.append(loss.item())
        # 优化器梯度清除
        optimizer_2.zero_grad()
        # 反向传播
        loss.backward()
        # 优化更新
        optimizer_2.step()
"""
with st.expander("创建模型并训练"):
    st.write("需要创建更加复杂的模型：")
    st.code(code_4, language='python')
    st.write("模型结构如下：")
    st.text(dnn_model)
    st.code(code_3, language='python')

batch_size_2 = st.slider("batch", 1, 100, 20, key="batch_size_2")

# 学习率
lr_2 = st.select_slider('learning_rate_b',options=[5e-5, 5e-4, 0.01, 0.02, 0.1, 0.2, 0.4], value=0.01, key="lr_2")

# epoch
epochs_2 = st.slider("epochs", 1, 200, 20, key="epochs_2")

# shuffle
shuffle_2 = st.toggle('打乱数据集', value=True, key="shuffle_2")

a_item = []
a_grad = []
loss_item = []
# 优化器
optimizer_2 = torch.optim.Adam(params=[p for n, p in dnn_model.named_parameters()], lr=lr_2)


progress_text = "training... Please wait."
training_bar = st.progress(0, text=progress_text)


for epoch in range(epochs_2):
    for d, t in get_batch_data(list(zip(x_sqr, y_sqr)), batch_size_2, shuffle_2):
        out = dnn_model(torch.unsqueeze(torch.tensor(d), dim=1))

        loss = torch.mean(torch.pow(out - torch.unsqueeze(torch.tensor(t),dim=1),2))
        loss_item.append(loss.item())
        # 优化器梯度清除
        optimizer_2.zero_grad()
        # 反向传播
        loss.backward()
        # a_grad.append(model.a.grad.item())
        # 优化更新
        optimizer_2.step()
        # a_item.append(model.a.item())
        
        training_bar.progress((epoch+1)/epochs_2, text=progress_text)
training_bar.empty()


st.line_chart(pd.DataFrame({"损失值loss": loss_item}), y="损失值loss")


final_text="""
### 结束语
在真正使用模型时，我们需要像上面一样，一步步创建模型吗？

是不用的，当前有很多开源社区，可以很方便的拿到一些模型。

比如开源社区[🤗 Tokenizers库](https://huggingface.co/) 和 [🤖 ModelScope](https://modelscope.cn/models)

"""


st.markdown(final_text)

