import torch
from torch import optim,nn 
from torchvision import models

import threading
import logging
import time
import matplotlib.pyplot as plt
import pandas as pd


log_path = 'logfile2.log'
logging.basicConfig(filename = log_path,
                    filemode = "w",
                    format = f"%(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

state = "load_model"
time_sleep = 0.1
EventMemRecord = threading.Event()

def mem_record():
    global state
    global time_sleep
    while(1):
        EventMemRecord.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},{state},{allocated_mem/1024**2},{max_mem/1024**2}")
        time.sleep(time_sleep)
        

def train():
    global state
    global time_sleep
    device = torch.device('cuda:0')

    state = "load_model"
    EventMemRecord.set()
    model = models.vgg16().to(device)
    EventMemRecord.clear()

    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    state = "train"
    EventMemRecord.set()
    img = torch.randn(1, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (1,)).to(device)
    EventMemRecord.clear()

    
    for i in range(3):
        state = "forward"
        time_sleep = 0.001
        EventMemRecord.set()
        for j in range(100):
            output = model(img)
        EventMemRecord.clear()
   
        loss = criterion(output, label)
        time_sleep = 0.0001
        state = "backward"
        EventMemRecord.set()
        loss.backward()
        EventMemRecord.clear()

        EventMemRecord.set()
        time_sleep = 0.0001
        state = "optimizer_step"
        optimizer.step()
        optimizer.zero_grad()
        EventMemRecord.clear()

def read_log(log_path):
    with open(log_path, 'r') as f:
        for line in f:
            yield  line.replace('"', '').strip().split(',')

# 使用matplotlib绘制内存使用情况
def draw_mem_usage(log_path):
    df = pd.DataFrame(read_log(log_path))
    df.columns = ['time', 'stage', 'mem_allocated', 'max_mem_reserved']
    df['time'] = df['time'].astype(float)
    df['mem_allocated'] = df['mem_allocated'].astype(float)
    df['max_mem_reserved'] = df['max_mem_reserved'].astype(float)
    df['stage'] = df['stage'].astype(str)
    #同一张图画出两条折线

    ax = df.plot(x='time', y='mem_allocated', color='red', label='mem_allocated')
    df.plot(x='time', y='max_mem_reserved', color='blue', label='max_mem_reserved', ax=ax)
    plt.show()

        
def main():
    # 创建线程读取mem写入log
    threads = []
    target_list = [mem_record, train]
    for target in target_list:
        t = threading.Thread(target=target)
        t.daemon = True if target != train else False
        t.start()
        threads.append(t)

    threads[-1].join() 

    # 读取log绘制内存使用情况
    draw_mem_usage(log_path)


if __name__=="__main__":
    main()
    
