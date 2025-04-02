from bayes_opt import BayesianOptimization, UtilityFunction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def bayesian_optimization_and_suggest(filepath,
                                      savepath,
                                      target_params="最大峰高",
                                      kind='ucb',
                                      kappa=10,
                                      n_points=5
                                      ):
    """
    贝叶斯优化参数推荐函数

    参数:
    filepath (str): 历史数据文件路径（Excel格式）
    savepath (str): 参数保存路径（Excel格式）
    kind (str): Utility函数类型，默认'ucb'
    kappa (float): 探索系数，默认10
    n_points (int): 建议点数量，默认5
    """

    # 加载数据
    df = pd.read_excel(filepath)

    # 数据标准化处理
    scaler = StandardScaler()
    df['标准化峰高'] = scaler.fit_transform(df[['最大峰高']])

    # 构建历史数据格式
    historical_data = []
    for _, row in df.iterrows():
        if pd.notnull(row[target_params]):
            # 如果没有删除的：
            historical_data.append({
                "params": {
                    "前驱体体积/ul": int(row['前驱体体积/ul']),
                    "旋涂速度/rpm": int(row['旋涂速度/rpm']),
                    "旋涂时间/s": int(row['旋涂时间/s']),
                    "旋涂加速度": int(row['旋涂加速度']),
                    "温度/°C": int(row['温度/°C']),
                    "退火时间/min": int(row['退火时间/min'])
                },
                "target": float(row[target_params])
            })

    # 参数边界和步长配置
    param_config = {
        "前驱体体积/ul": {"bounds": (10, 100), "step": 10},
        "旋涂速度/rpm": {"bounds": (500, 5000), "step": 500},
        "旋涂时间/s": {"bounds": (30, 180), "step": 30},
        "旋涂加速度": {"bounds": (500, 5000), "step": 500},
        "温度/°C": {"bounds": (80, 160), "step": 20},
        "退火时间/min": {"bounds": (5, 60), "step": 5}
    }

    # 初始化优化器
    optimizer = BayesianOptimization(
        f=None,
        pbounds={k: v["bounds"] for k, v in param_config.items()},
        verbose=0,
        allow_duplicate_points=False,
        random_state=42
    )

    # 注册历史数据
    for data in historical_data:
        optimizer.register(
            params=data["params"],
            target=data["target"]
        )

    # 生成建议点的内部函数
    def _suggest_points(n_points, max_attempts=1000):
        suggestions = []
        existing_params = set(tuple(int(x) for x in p) for p in optimizer._space.params)

        attempt_count = 0
        while len(suggestions) < n_points and attempt_count < max_attempts:
            attempt_count += 1
            try:
                utility = UtilityFunction(kind=kind, kappa=kappa)
                raw_params = optimizer.suggest(utility)

                # 参数对齐处理
                aligned_params = {}
                for param, config in param_config.items():
                    min_val, max_val = config["bounds"]
                    step = config["step"]
                    aligned_value = round(raw_params[param] / step) * step
                    aligned_params[param] = int(np.clip(aligned_value, min_val, max_val))

                # 检查唯一性
                param_tuple = tuple(aligned_params.values())
                if param_tuple not in existing_params:
                    suggestions.append(aligned_params)
                    existing_params.add(param_tuple)
            except Exception as e:
                continue
        return suggestions[:n_points]

    # 生成建议参数
    suggestions = _suggest_points(n_points)

    # 转换为DataFrame并添加实验序号
    df_suggestions = pd.DataFrame(suggestions)

    # 确定起始序号
    start_num = df['实验序号'].max() + 1 if '实验序号' in df.columns else 1
    df_suggestions.insert(0, '实验序号', range(start_num, start_num + len(df_suggestions)))

    # 列顺序调整
    column_order = [
        '实验序号',
        '前驱体体积/ul',
        '旋涂速度/rpm',
        '旋涂时间/s',
        '旋涂加速度',
        '温度/°C',
        '退火时间/min'
    ]
    df_suggestions = df_suggestions[column_order]

    # 保存结果
    df_suggestions.to_excel(savepath, index=False, engine='openpyxl')
    print(f"优化参数已保存至：{savepath}")


if __name__=="__main__":
    # 使用示例
    bayesian_optimization_and_suggest(
        filepath='initial_data_with_features.xlsx',
        savepath='BO_Round1.xlsx',
        kind='ucb',
        kappa=10,
        n_points=25,

    )