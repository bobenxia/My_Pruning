import pandas as pd

excel_path = "model_data.xlsx"
df1 = pd.DataFrame({'Model': ['Resnet-18'], 'Top1-acc(%)': [92.34],
                   '效': [60, 84, 98, 91], '备注': ['不及格', '良好', '最佳', '优秀']})
print(df1)

df1.to_excel(excel_path)
df2 = pd.read_excel(excel_path, index_col=0)
print(df2.columns.values)

df2.loc[df2.index.max()+1] = ['x', '23', '23', '23']
print(df2)


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
