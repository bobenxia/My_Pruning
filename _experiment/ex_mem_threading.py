import torch
from torch import optim,nn 
from torchvision import models

import threading
import logging
import time
import csv
import io

EventLoadModel = threading.Event()
EventLoadData = threading.Event()
EventForward = threading.Event()
EventBackward = threading.Event()
EventOptimStep = threading.Event()


logging.basicConfig(filename = "logfile2.log",
                    filemode = "w",
                    format = f"%(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def mem_load_model():
    while(1):
        EventLoadModel.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},load model,{allocated_mem/1024**2},{max_mem/1024**2}")
        time.sleep(0.01)
        

def mem_load_data():
    while(1):
        EventLoadData.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},load data,{allocated_mem/1024**2},{max_mem/1024**2}")
        time.sleep(0.01)

def mem_forward():
    while(1):
        EventForward.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},forward,{allocated_mem/1024**2},{max_mem/1024**2}")
        time.sleep(0.01)

def mem_backward():
    while(1):
        EventBackward.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},backward,{allocated_mem/1024**2},{max_mem/1024**2}")
        # time.sleep(0.0001)

def mem_optim_step():
    while(1):
        EventOptimStep.wait()
        allocated_mem = torch.cuda.memory_allocated()
        max_mem = torch.cuda.max_memory_reserved()
        logging.info(f"{time.time()},optim step,{allocated_mem/1024**2},{max_mem/1024**2}")
        # time.sleep(0.0001)


def train():
    device = torch.device('cuda:0')

    EventLoadModel.set()
    model = models.vgg16().to(device)
    EventLoadModel.clear()

    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    EventLoadData.set()
    img = torch.randn(1, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (1,)).to(device)
    EventLoadData.clear()

    
    for i in range(3):
        EventForward.set()
        for j in range(100):
            output = model(img)
        EventForward.clear()
        # time.sleep(1)

        

        loss = criterion(output, label)
        EventBackward.set()
        loss.backward()
        EventBackward.clear()
        # time.sleep(1)

        

        EventOptimStep.set()
        optimizer.step()
        optimizer.zero_grad()
        EventOptimStep.clear()
        # time.sleep(1)

        


def main():
    threads = []
    target_list = [mem_load_model, mem_load_data, mem_forward, mem_backward, mem_optim_step, train]
    for target in target_list:
        t = threading.Thread(target=target)
        t.daemon = True if target != train else False
        t.start()
        threads.append(t)

    threads[-1].join() 


if __name__=="__main__":
    main()
