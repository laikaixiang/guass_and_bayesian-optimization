import pandas as pd
import os


def extract_features_and_merge(fit_params_path, process_data_path, output_path):
    """
    从拟合参数和工艺数据中提取特征并合并

    参数:
        fit_params_path: 拟合参数CSV文件路径
        process_data_path: 工艺数据Excel文件路径
        output_path: 输出文件路径

    返回:
        合并后的DataFrame
    """
    # 1. 读取数据
    fit_params = pd.read_csv(fit_params_path)
    process_data = pd.read_excel(process_data_path)

    # 2. 检查输出文件是否存在
    if os.path.exists(output_path):
        existing_data = pd.read_excel(output_path)
        last_exp_id = existing_data['实验序号'].max()
        # 调整新数据的实验序号
        process_data['实验序号'] = process_data['实验序号'] + last_exp_id
    else:
        existing_data = None

    # 3. 初始化特征字典
    features = {}
    for exp_id in process_data['实验序号']:
        features[exp_id] = {
            '最大峰高': None,
            '最大峰高_std': None,
        }

    # 4. 按实验序号分组处理
    grouped = fit_params.groupby(fit_params.columns[0])

    for exp_id, group in grouped:
        # 调整实验序号以匹配process_data
        adjusted_exp_id = exp_id + (last_exp_id if existing_data is not None else 0)

        # 提取峰高相关特征
        peak_cols = [col for col in group.columns if '峰高' in col]

        # 计算最大峰高
        max_peak_value = group[peak_cols].max().max()
        features[adjusted_exp_id]['最大峰高'] = max_peak_value

        # 提取最大峰对应的标准差
        max_peak_pos = group[peak_cols].stack().idxmax()
        peak_num = max_peak_pos[1][-1]
        std_col = f'标准差{peak_num}'
        features[adjusted_exp_id]['最大峰高_std'] = group.loc[max_peak_pos[0], std_col]

        # 其他特征
        # 添加示例示例: 计算两个峰高的比值
        # if len(peak_cols) >= 2:
        #     features[exp_id]['峰高比'] = group[peak_cols[0]].mean() / group[peak_cols[1]].mean()


    # 5. 转换为DataFrame并合并
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = '实验序号'
    merged_df = process_data.merge(features_df, on='实验序号', how='left')

    # 6. 保存结果（追加或新建）
    if existing_data is not None:
        final_df = pd.concat([existing_data, merged_df], ignore_index=False)
    else:
        final_df = merged_df

    final_df.to_excel(output_path, index=False)

    return final_df


# 使用示例
if __name__ == "__main__":
    # 基本用法
    df = extract_features_and_merge(
        fit_params_path="拟合参数第一轮.csv",
        process_data_path="贝叶斯优化/数据备份/initial_data.xlsx",
        output_path="initial_data_with_features.xlsx"
    )