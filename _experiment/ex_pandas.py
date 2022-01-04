import pandas as pd

excel_path = "model_data.xlsx"
# df1 = pd.DataFrame({
#     'Model': ['Resnet-18'],
#     'If_base': ['True'],
#     'Top1-acc(%)': [92.34],
#     'Top5-acc(%)': [62.34],
#     'G_Mac': [112],
#     'G_Mac percent(%)': [' '],
#     'M_params': [3.1],
#     'M_params percent(%)': [' '],
#     'ms_cpu_time': [3484.94],
#     'ms_gpu_time': [2.634],
#     'ms_gpu_time percent(%)': [' '],
#     'M_gpu_mem': [1508.625],
#     'M_gpu_mem percent(%)': [' ']
# })
# df1 = pd.DataFrame(columns=[
#     'Model', 'If_base', 'Top1-acc(%)', 'Top5-acc(%)', 'G_Mac', 'G_Mac percent(%)', 'M_params',
#     'M_params percent(%)', 'ms_cpu_time', 'ms_gpu_time', 'ms_gpu_time percent(%)', 'M_gpu_mem',
#     'M_gpu_mem percent(%)', 'Strategy'
# ])
# print(df1)

# df1.to_excel(excel_path)

# df2 = pd.read_excel(excel_path, index_col=0)
# print(df2.columns.values)

# df2.loc[df2.index.max() + 1] = ['resnet-34',' ','93.22', 'None','3.680',' ',  '21.798',' ', '3782.843', '2.636', ' ','1000',' ']
# print(df2)
# df2.to_excel(excel_path)

# import pandas as pd

# list_data = [
#     ['a', 'b', 'c', ],
#     ['a1', 'b1', 'c1'],
# ]  # 使用二维数组
# df = pd.DataFrame(data=list_data)
# df =pd.DataFrame({'名字':['老王','刘'],'工资':[500,700],'效':[60,84]})
# print(df )
# df.loc[df.index.max() + 1] = ['a2', 'b2', 'c2']
# print(df )
