import pandas as pd

excel_path = "model_data.xlsx"
df1 = pd.DataFrame(columns=[
    'Model', 'If_base', 'Top1-acc(%)', 'Top5-acc(%)', 'G_Mac', 'G_Mac percent(%)', 'M_params', 'M_params percent(%)',
    'ms_cpu_time', 'ms_gpu_time', 'ms_gpu_time percent(%)', 'M_gpu_mem', 'M_gpu_mem percent(%)', 'Strategy'
])
print(df1)

df1.to_excel(excel_path)
