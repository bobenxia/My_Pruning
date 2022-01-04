import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def calcu_percent(param_a, param_b):
    return round((param_a - param_b) / param_a * 100.0, 3)


def read_excel_and_write(excel_path: str, new_model_infor: dict):
    df = pd.read_excel(excel_path, index_col=0)

    # Get the information of the model with the same name from the table
    model_df = df.query("Model == @new_model_infor['Model']")
    if model_df.empty:
        print("There is no model with the same name in the table")
        # Whether the model_info contains the information of the basic model
        if new_model_infor['If_base'] != 'True':
            raise Exception(
                "Please record the basic model with the same name in the table first(when the keyword 'If_base' is 'True', the model is the base model)"
            )
        else:
            new_model_infor['G_Mac percent(%)'] = new_model_infor['M_params percent(%)'] = new_model_infor[
                'ms_gpu_time percent(%)'] = new_model_infor['M_gpu_mem percent(%)'] = ' '
    else:
        if new_model_infor['If_base'] in ['True', 'TRUE', 'true', 1]:
            raise Exception(
                "Only one base model is allowed for each model in each table(when the keyword 'If_base' is 'True', the model is the base model)"
            )
        else:
            base_mode_df = model_df.query('(If_base == 1)').iloc[0]
            new_model_infor['G_Mac percent(%)'] = calcu_percent(base_mode_df['G_Mac'], new_model_infor['G_Mac'])
            new_model_infor['M_params percent(%)'] = calcu_percent(base_mode_df['M_params'],
                                                                   new_model_infor['M_params'])
            new_model_infor['ms_gpu_time percent(%)'] = calcu_percent(base_mode_df['ms_gpu_time'],
                                                                      new_model_infor['ms_gpu_time'])
            new_model_infor['M_gpu_mem percent(%)'] = calcu_percent(base_mode_df['M_gpu_mem'],
                                                                    new_model_infor['M_gpu_mem'])

    wirte_index = (df.index.max() + 1) if not df.empty else 0
    df.loc[wirte_index] = [
        new_model_infor['Model'], new_model_infor['If_base'], new_model_infor['Top1-acc(%)'],
        new_model_infor['Top5-acc(%)'], new_model_infor['G_Mac'], new_model_infor['G_Mac percent(%)'],
        new_model_infor['M_params'], new_model_infor['M_params percent(%)'], new_model_infor['ms_cpu_time'],
        new_model_infor['ms_gpu_time'], new_model_infor['ms_gpu_time percent(%)'], new_model_infor['M_gpu_mem'],
        new_model_infor['M_gpu_mem percent(%)'], new_model_infor['Strategy']
    ]
    new_model_df = df.query("Strategy == @new_model_infor['Strategy'] and Model == @new_model_infor['Model']")
    print(f"Model information with the same name in the table: \n {model_df} \n")
    print(f"New model information:\n {new_model_df} \n")

    str = input("Whether to write model information into Excel(y or n)：")
    if str in ['y', 'Y', 'yes']:
        df.to_excel(excel_path)

    return None


if __name__ == "__main__":
    # import torch
    # import torchvision.models as models

    # from tools.print_model_info import get_model_infor_and_print

    # model = models.resnet34()
    # data = torch.randn(1, 3, 224, 224)
    # gpu_id = 4

    # model_infor = get_model_infor_and_print(model, data, gpu_id)
    # print(model_infor)

    # model_infor['Model'] = 'resnet-34'
    # model_infor['Top1-acc(%)'] = 91.11
    # model_infor['Top5-acc(%)'] = 93.23
    # model_infor['If_base'] = 'True'

    # print(model_infor)

    # model_infor = {
    #     'G_Mac': 3.676,
    #     'M_params': 21.798,
    #     'ms_cpu_time': 3671.921,
    #     'ms_gpu_time': 2.633,
    #     'M_gpu_mem': 1508.625,
    #     'Model': 'resnet-12',
    #     'Top1-acc(%)': 91.11,
    #     'Top5-acc(%)': 93.23,
    #     'If_base': 'True',
    #     'Strategy': ' '
    # }
    model_infor = {
        'G_Mac': 2.676,
        'M_params': 14.798,
        'ms_cpu_time': 3671.921,
        'ms_gpu_time': 2.033,
        'M_gpu_mem': 1208.625,
        'Model': 'resnet-12',
        'Top1-acc(%)': 91.11,
        'Top5-acc(%)': 93.23,
        'If_base': 'False',
        'Strategy': '[0.2, 0.3, ..., 0.5]'
    }
    excel_path = "model_data.xlsx"
    read_excel_and_write(excel_path, model_infor)
