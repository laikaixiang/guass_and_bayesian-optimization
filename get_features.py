import pandas as pd
from typing import List, Dict, Optional


def extract_features_and_merge(
        fit_params_path: str,
        process_data_path: str,
        output_path: str,
        additional_features: Optional[List[str]] = None,
        max_peak_std: bool = False
) -> pd.DataFrame:
    """
    从拟合参数和工艺数据中提取特征并合并

    参数:
        fit_params_path: 拟合参数CSV文件路径
        process_data_path: 工艺数据Excel文件路径
        output_path: 输出文件路径
        additional_features: 需要提取的其他特征列表(如['均值1', '标准差1']等)
        max_peak_std: 是否提取最大峰对应的标准差

    返回:
        合并后的DataFrame
    """
    # 读取数据
    fit_params = pd.read_csv(fit_params_path)
    process_data = pd.read_excel(process_data_path)

    # 初始化特征字典
    features = {}

    # 按实验序号分组
    grouped = fit_params.groupby(fit_params.columns[0])

    # 提取最大峰高及其对应的标准差
    for exp_id, group in grouped:
        # 找出所有峰高列
        peak_cols = [col for col in group.columns if '峰高' in col]

        # 找出最大峰高及其位置
        max_peak_value = group[peak_cols].max().max()
        max_peak_pos = group[peak_cols].stack().idxmax()  # (row, column)

        # 存储最大峰高
        features[exp_id] = {'最大峰高': max_peak_value}

        # 如果需要最大峰对应的标准差
        if max_peak_std:
            # 获取对应的峰编号(峰高1或峰高2)
            peak_num = max_peak_pos[1][-1]  # 提取峰高列名的最后一个字符(1或2)
            std_col = f'标准差{peak_num}'
            max_peak_std_value = group.loc[max_peak_pos[0], std_col]
            features[exp_id]['最大峰高_std'] = max_peak_std_value

        # 提取其他指定特征
        if additional_features:
            for feat in additional_features:
                # 计算该特征在所有行中的平均值
                features[exp_id][feat] = group[feat].mean()

    # 转换为DataFrame
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = '实验序号'

    # 合并工艺数据
    merged_df = process_data.merge(features_df, left_on='实验序号', right_index=True, how='left')

    # 保存结果
    merged_df.to_excel(output_path, index=False)

    return merged_df


# 使用示例
if __name__ == "__main__":
    # 基本用法 - 只提取最大峰高
    df = extract_features_and_merge(
        fit_params_path="拟合参数第一轮.csv",
        process_data_path="initial_data.xlsx",
        output_path="initial_data_with_features.xlsx"
    )

    # # 高级用法 - 提取最大峰高及其标准差，以及其他特征
    # df = extract_features_and_merge(
    #     fit_params_path="拟合参数第一轮.csv",
    #     process_data_path="initial_data.xlsx",
    #     output_path="initial_data_with_features_advanced.xlsx",
    #     additional_features=['均值1', '均值2'],
    #     max_peak_std=True
    # )