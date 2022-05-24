import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

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


if __name__=="__main__":
    log_path = 'logfile2.log'
    draw_mem_usage(log_path)