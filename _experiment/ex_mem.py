# # %%
# import torch

# model = torch.nn.Linear(1024,1024, bias=False).cuda() 
# optimizer = torch.optim.AdamW(model.parameters())
# mem1 = torch.cuda.memory_allocated()
# print(mem1)
# # %%
# inputs = torch.tensor([1.0]*1024).cuda() # shape = (1024)
# mem2 = torch.cuda.memory_allocated()
# print(mem2)
# print(mem2-mem1)
# outputs = model(inputs) # shape = (1024)
# mem3 = torch.cuda.memory_allocated()
# print(mem3)
# print(mem3-mem2)
# # %%
# loss = sum(outputs) # shape = (1)
# mem4 = torch.cuda.memory_allocated()
# print(mem4)
# print(mem4-mem3)
# # %%
# # 保存梯度的值
# loss.backward()
# mem5 = torch.cuda.memory_allocated()
# print(mem5)
# print(mem5-mem4)
# # %%
# optimizer.step()
# mem6 = torch.cuda.memory_allocated()
# print(mem6)
# print(mem6-mem5)
# # %%
import torch


# 模型初始化
linear1 = torch.nn.Linear(1024,1024, bias=False).cuda() # + 4194304
print(torch.cuda.memory_allocated())
linear2 = torch.nn.Linear(1024, 1, bias=False).cuda() # + 4096
print('模型初始化： ',torch.cuda.memory_allocated())

# 输入定义
inputs = torch.tensor([[1.0]*1024]*1024).cuda() # shape = (1024,1024) # + 4194304
print('输入定义： ',torch.cuda.memory_allocated())

# 前向传播
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304 + 512
print('第一次前向传播： ',torch.cuda.memory_allocated())

# 后向传播
loss.backward() # memory - 4194304 + 4194304 + 4096
print('第一次后向传播： ',torch.cuda.memory_allocated())

# 再来一次~
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
print('第二次前向传播： ',torch.cuda.memory_allocated())
loss.backward() # memory - 4194304
print('第二次反向传播： ',torch.cuda.memory_allocated())


# 再来一次~
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
print('第二次前向传播： ',torch.cuda.memory_allocated())
loss.backward() # memory - 4194304
print('第二次反向传播： ',torch.cuda.memory_allocated())
