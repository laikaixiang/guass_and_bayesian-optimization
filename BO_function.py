from bayes_opt import BayesianOptimization, UtilityFunction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def bayesian_optimization_and_suggest(filepath,
                                      savepath,
                                      target_params="最大峰高",
                                      kind='ucb',
                                      kappa=10,
                                      n_points=5,
                                      param_config=None
                                      ):
    """
    贝叶斯优化参数推荐函数

    参数:
    filepath (str): 历史数据文件路径（Excel格式）
    savepath (str): 参数保存路径（Excel格式）
    target_params (str): 目标参数列名
    kind (str): Utility函数类型，默认'ucb'
    kappa (float): 探索系数，默认10
    n_points (int): 建议点数量，默认5
    param_config (dict): 参数配置，格式为 {
        '参数名': {'bounds': (min, max), 'step': step, 'dtype': 'int'/'float'}
    }

    返回:
    None

    异常:
    ValueError: 当数据文件缺少必要的列时抛出
    """

    # 加载数据
    df = pd.read_excel(filepath)

    # 默认参数配置
    # param_config = {
    #     "前驱体体积/ul": {"bounds": (10, 100), "step": 10},
    #     "旋涂速度/rpm": {"bounds": (500, 5000), "step": 500},
    #     "旋涂时间/s": {"bounds": (30, 180), "step": 30},
    #     "旋涂加速度": {"bounds": (500, 5000), "step": 500},
    #     "温度/°C": {"bounds": (80, 160), "step": 20},
    #     "退火时间/min": {"bounds": (5, 60), "step": 5}
    # }
    default_param_config = {
        "前驱体体积/ul": {"bounds": (10, 60), "step": 5, "dtype": "int"},
        "旋涂速度/rpm": {"bounds": (1000, 5000), "step": 200, "dtype": "int"},
        "旋涂时间/s": {"bounds": (10, 60), "step": 5, "dtype": "int"},
        "旋涂加速度": {"bounds": (1000, 5000), "step": 200, "dtype": "int"},
        "退火时间/min": {"bounds": (1, 10), "step": 1, "dtype": "int"},
        "添加剂": {"bounds": (1, 12), "step": 1, "dtype": "int"},
        "添加量%": {"bounds": (5, 20), "step": 5, "dtype": "int"}
    }

    # 使用用户提供的参数配置或默认配置
    param_config = param_config or default_param_config

    # 检查目标参数列是否存在
    if target_params not in df.columns:
        raise ValueError(f"目标参数列 '{target_params}' 不存在于数据文件中")

    # 检查所有参数列是否存在
    missing_columns = [param for param in param_config.keys() if param not in df.columns]
    if missing_columns:
        raise ValueError(f"数据文件缺少以下参数列: {', '.join(missing_columns)}")

    # 数据标准化处理（如果目标参数存在）
    if target_params in df.columns and pd.notnull(df[target_params]).any():
        scaler = StandardScaler()
        df['标准化目标'] = scaler.fit_transform(df[[target_params]])

    # 构建历史数据格式
    historical_data = []
    for _, row in df.iterrows():
        if pd.notnull(row[target_params]):
            params = {}
            for param_name in param_config.keys():
                params[param_name] = row[param_name]

            historical_data.append({
                "params": params,
                "target": float(row[target_params])
            })

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
                    dtype = config.get("dtype", "int")

                    # 对齐到最近的步长倍数
                    aligned_value = round(raw_params[param] / step) * step

                    # 确保在边界内
                    aligned_value = np.clip(aligned_value, min_val, max_val)

                    # 转换为指定类型
                    if dtype == "int":
                        aligned_params[param] = int(aligned_value)
                    else:
                        aligned_params[param] = float(aligned_value)

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

    # 保存结果
    df_suggestions.to_excel(savepath, index=False, engine='openpyxl')
    print(f"优化参数已保存至：{savepath}")


if __name__ == "__main__":
    try:
        # 使用示例
        # 使用示例
        bayesian_optimization_and_suggest(
            filepath='training_data/initial_data_with_features.xlsx',
            savepath='newProcesses/BO_Round1.xlsx',
            target_params="最大峰高",
            kind='ucb',
            kappa=10,
            n_points=20,

        )
    except ValueError as e:
        print(f"错误: {e}")
