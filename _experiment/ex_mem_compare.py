from cProfile import label
import pynvml
import torch
from torch import optim, nn
from torchvision import models
import inspect
from torchvision import models
from gpu_mem_track import MemTracker  # 引用显存跟踪代码
import time

# @profile
def test():
    torch.cuda._lazy_init()
    device = torch.device('cuda:0')
    model = models.resnet34().to(device)  # 导入VGG19模型并且将数据转到显存中
    # model = torch.nn.Conv2d(1, 1, 1).to(device)

def test2():
    device = torch.device('cuda:0')
    frame = inspect.currentframe()     
    gpu_tracker = MemTracker(frame)      # 创建显存检测对象
    gpu_tracker.track()                  # 开始检测
    gpu_tracker.track()
    cnn = models.vgg19(pretrained=True).to(device)  # 导入VGG19模型并且将数据转到显存中
    gpu_tracker.track()

def test3():
    mem1 = torch.cuda.memory_allocated()
    device = torch.device('cuda:0')
    model = models.vgg16().to(device)
    mem2 = torch.cuda.memory_allocated()
    print("model  \t", (mem2 - mem1)/1024**2)


def test4():
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    mem1 = torch.cuda.memory_allocated()
    print("mem1      \t", mem1/1024**2)
    time.sleep(5)

    device = torch.device('cuda:0')
    model = models.vgg16().to(device)
    # model2 = models.resnet50().to(device)
    mem2 = torch.cuda.memory_allocated()
    print("model     \t", (mem2 - mem1)/1024**2)
    print("mem2      \t", mem2/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem   \t", max_mem/1024**2)

    time.sleep(5)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    img = torch.randn(1, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (1,)).to(device)
    mem3 = torch.cuda.memory_allocated()
    print("img + label  \t", (mem3 - mem2)/1024**2)
    print("mem3        \t", mem3/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    
    # print("optimizer.zero_grad  \t", (mem6 - mem5)/1024**2)

    for i in range(100):
        output = model(img)
    mem4 = torch.cuda.memory_allocated()
    print("output  \t", (mem4 - mem3)/1024**2)
    print("mem4    \t", mem4/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    loss = criterion(output, label)
    loss.backward()
    mem5 = torch.cuda.memory_allocated()
    print("loss.backward\t", (mem5 - mem4)/1024**2)
    print("mem5        \t", mem5/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    optimizer.step()
    mem6 = torch.cuda.memory_allocated()
    print("optimizer.step\t", (mem6 - mem5)/1024**2)
    print("mem6      \t", mem6/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    optimizer.zero_grad()
    mem7 = torch.cuda.memory_allocated()
    print("optimizer.zero_grad\t", (mem7 - mem6)/1024**2)
    print("mem7      \t", mem7/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    

    for i in range(100):
        output = model(img)
    mem8 = torch.cuda.memory_allocated()
    print("output  \t", (mem8 - mem7)/1024**2)
    print("mem8    \t", mem8/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    loss = criterion(output, label)
    loss.backward()
    mem9 = torch.cuda.memory_allocated()
    print("loss.backward\t", (mem9 - mem8)/1024**2)
    print("mem9        \t", mem9/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    optimizer.step()
    mem10 = torch.cuda.memory_allocated()
    print("optimizer.step\t", (mem10 - mem9)/1024**2)
    print("mem10      \t", mem10/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    time.sleep(5)

    optimizer.zero_grad()
    mem11 = torch.cuda.memory_allocated()
    print("optimizer.zero_grad\t", (mem11 - mem10)/1024**2)
    print("mem11      \t", mem11/1024**2)
    max_mem = torch.cuda.max_memory_allocated()
    print("max_mem  \t", max_mem/1024**2)

    print("total        \t", (mem11 - mem1)/1024**2)


def print_mem():
    mem = torch.cuda.memory_allocated()
    print("mem  \t", mem/1024**2)

def print_mem_reserve():
    mem = torch.cuda.memory_reserved()
    print("mem_reserve\t", mem/1024**2)

def print_mem_max():
    mem = torch.cuda.max_memory_allocated()
    print("mem_max  \t", mem/1024**2)

def print_mem_reserve_max():
    mem = torch.cuda.max_memory_reserved()
    print("mem_reserve_max\t", mem/1024**2)

def empty_mem():
    torch.cuda.empty_cache()

def print_func():
    print_mem()
    print_mem_max()
    print_mem_reserve()
    print_mem_reserve_max()
    print("\n")
    # empty_mem()
    # print(torch.cuda.memory_summary())


def test5():
    print_func()

    print("model:+存储模型的空间")
    device = torch.device('cuda:0')
    model = models.vgg13().to(device)
    print_func()

    print("optimizer:")
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    img = torch.randn(1, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (1,)).to(device)
    print_func()

    for i in range(2):
        print("output:+存储中间结果和最终结果的空间")
        for i in range(100):
            output = model(img)
        print_func()

        print("loss.backward:+存储梯度的空间 - 存储中间结果和最终结果的空间")
        loss = criterion(output, label)
        loss.backward()
        print_func()

        print("optimizer.step:+存储优化器的空间")
        optimizer.step()
        print_func()

        print("optimizer.zero_grad:")
        optimizer.zero_grad()
        print_func()
        
        time.sleep(5)


def test6():
    mem0 = torch.cuda.memory_allocated()
    device = torch.device('cuda:0')
    model = models.vgg16().to(device)
    mem1 = torch.cuda.memory_allocated()
    mem_model = mem1 - mem0


    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    mem2 = torch.cuda.memory_allocated()
    img = torch.randn(1, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (1,)).to(device)
    mem3 = torch.cuda.memory_allocated()
    mem_data = mem3 - mem2

    for i in range(2):
        mem4 = torch.cuda.memory_allocated()
        for i in range(100):
            output = model(img)
        mem5 = torch.cuda.memory_allocated()
        mem_output = mem5 - mem4

        mem6 = torch.cuda.memory_allocated()
        loss = criterion(output, label)
        loss.backward()
        mem7 = torch.cuda.memory_allocated()
        mem_loss_backward = mem7 - mem6 + mem_output

        mem8 = torch.cuda.memory_allocated()
        mem8_1 = torch.cuda.max_memory_reserved()
        optimizer.step()
        mem9 = torch.cuda.memory_allocated()
        mem9_1 = torch.cuda.max_memory_reserved()
        mem_optimizer_step = mem9 - mem8
        mem_optimizer_step_1 = mem9_1 - mem8_1

        optimizer.zero_grad()

        print("mem_model\t", mem_model/1024**2)
        print("mem_data\t", mem_data/1024**2)
        print("mem_output\t", mem_output/1024**2)
        print("mem_loss_backward\t", mem_loss_backward/1024**2)
        print("mem_optimizer_step\t", mem_optimizer_step/1024**2)
        print("mem_optimizer_step_1\t", mem_optimizer_step_1/1024**2)
        print_mem_max()
        print_mem_reserve_max()
        print("total\t", (mem_model + mem_data + mem_output + mem_loss_backward + mem_optimizer_step_1)/1024**2)
        time.sleep(5)

if __name__ == '__main__':
    test5()

