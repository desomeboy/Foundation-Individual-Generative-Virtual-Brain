import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import pandas as pd
import numpy as np
import torch
from npi import (
    device, ensure_dir,
    ANN_MLP, ANN_Transformer,
    load_model, analyze_single_patient, load_patient_data
)
from npi.config import *
ROI_NUM = 360

# 配置参数 (需与训练时保持一致)
TRAINING_PARAMS = {
    'steps': DEFAULT_STEPS,        # 时间步长 (30)
    'skip_first': DEFAULT_SKIP_FIRST,  # 跳过的初始时间点 (30)
    'model_type': 'transformer',
    'd_model': 768,
    'nhead': DEFAULT_NHEAD,
    'num_layers': 2,
    'num_cross_layers': 1,
    'use_last_token': False
}

def clean_check_name(check_name):
    return str(check_name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")

def get_target_label(target_str):
    target_str = str(target_str).lower()
    if '左侧' in target_str or 'left' in target_str:
        side = 'left'
    elif '右侧' in target_str or 'right' in target_str:
        side = 'right'
    else:
        return None

    if 'stn' in target_str:
        target = 'stn'
    elif 'gpi' in target_str:
        target = 'gpi'
    else:
        return None

    return f"{side}_{target}"

def load_and_predict(check_before, check_after, base_dir, data_dir):
    model_path = f"{base_dir}/{check_after}/patient_models/model_{check_after}.pth"
    csv_path = f"{data_dir}/{check_before}.csv"

    if not os.path.exists(model_path):
        print(f"  警告: 模型文件不存在 - {model_path}")
        return None
    if not os.path.exists(csv_path):
        print(f"  警告: CSV文件不存在 - {csv_path}")
        return None

    try:
        inputs, targets, labels, num_scans = load_patient_data(
            patient_file=csv_path,
            steps=TRAINING_PARAMS['steps'],
            skip_first=TRAINING_PARAMS['skip_first'],
            dataset_type='OTHER',
            label_map=None
        )
        if inputs is None or len(inputs) == 0:
            print(f"  警告: 数据加载为空 - {csv_path}")
            return None
    except Exception as e:
        print(f"  警告: 数据加载失败 ({csv_path}): {str(e)}")
        return None

    input_dim = TRAINING_PARAMS['steps'] * ROI_NUM
    model = ANN_Transformer(
        input_dim=input_dim,
        steps=TRAINING_PARAMS['steps'],
        roi_num=ROI_NUM,
        d_model=TRAINING_PARAMS['d_model'],
        nhead=TRAINING_PARAMS['nhead'],
        num_layers=TRAINING_PARAMS['num_layers'],
        dim_feedforward=2 * TRAINING_PARAMS['d_model'],
        dropout=0.1,
        use_layernorm=True,
        use_last_token=TRAINING_PARAMS['use_last_token'],
        num_labels=4,
        num_cross_layers=TRAINING_PARAMS['num_cross_layers']
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"  警告: 模型加载失败 ({model_path}): {str(e)}")
        return None

    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs_pred = model(inputs_tensor, labels=labels_tensor)

    bias = outputs_pred.cpu().numpy() - targets
    avg_bias = np.mean(bias, axis=0)  # shape: (360,)
    return avg_bias

def main():
    xlsx_path = "/ailab/user/dusiyuan/code/Brain/EC/ruijin/fmri_symptoms.xlsx"
    base_dir = "/ailab/user/dusiyuan/code/Brain/EC/ruijin_10.10"
    output_base = "/ailab/user/dusiyuan/code/Brain/EC/ruijin/delta_model_output"
    data_dir = '/ailab/group/medai-share/syDu/ruijin/10.10/output_csv'

    ensure_dir(output_base)
    target_labels = ['left_stn', 'left_gpi', 'right_stn', 'right_gpi']
    for label in target_labels:
        ensure_dir(os.path.join(output_base, label))

    print("解析病人数据...")
    df = pd.read_excel(xlsx_path)

    # 填充关键列
    df['序号'] = df['序号'].ffill()
    df['日期'] = df['日期'].ffill()
    df['姓名'] = df['姓名'].ffill()
    df['靶点位置'] = df['靶点位置'].ffill()

    treatment_events = []
    grouped = df.groupby(['序号', '日期', '姓名', '靶点位置'])

    for (id, date, name, target), group in grouped:
        if len(group) != 2:
            continue
        if not ({'前', '后'} <= set(group['治疗状态'].values)):
            continue

        row_before = group[group['治疗状态'] == '前'].iloc[0]
        row_after = group[group['治疗状态'] == '后'].iloc[0]

        check_before = clean_check_name(row_before['Check'])
        check_after = clean_check_name(row_after['Check'])
        score_before = row_before['总分']
        score_after = row_after['总分']

        target_label = get_target_label(target)
        if not target_label:
            continue

        treatment_events.append({
            'check_before': check_before,
            'check_after': check_after,
            'score_before': score_before,
            'score_after': score_after,
            'name': clean_check_name(str(name)),
            'target_label': target_label
        })

    print(f"找到 {len(treatment_events)} 个有效治疗事件")

    for i, event in enumerate(treatment_events):
        print(f"\n处理事件 {i+1}/{len(treatment_events)}:")
        print(f"  姓名: {event['name']}")
        print(f"  治疗前: {event['check_before']} (总分: {event['score_before']})")
        print(f"  治疗后: {event['check_after']} (总分: {event['score_after']})")
        print(f"  靶点: {event['target_label']}")

        avg_bias = load_and_predict(
            event['check_before'],
            event['check_after'],
            base_dir,
            data_dir
        )

        if avg_bias is not None:
            # 构建文件名：包含姓名、前后分数、check_before（唯一标识）
            safe_name = event['name']
            filename = f"{safe_name}_{event['score_before']}_to_{event['score_after']}_{event['check_before']}_bias.csv"
            output_path = os.path.join(output_base, event['target_label'], filename)

            # 保存为 (360,) 的单列 CSV，带 ROI 索引
            pd.DataFrame(
                avg_bias,
                columns=['avg_bias'],
                index=[f'ROI_{i+1}' for i in range(ROI_NUM)]
            ).to_csv(output_path, index=True, header=True)

            print(f"  保存成功: {output_path}")
        else:
            print("  处理失败，跳过此事件")

    print("\n分析完成！所有单次治疗偏差已保存。")

if __name__ == "__main__":
    main()