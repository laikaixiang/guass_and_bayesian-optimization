import pytz
from data_processing import analyze_spectral_uniformity
from get_features import extract_features_and_merge
from BO_function import bayesian_optimization_and_suggest
from guass_function import guassFitData


if __name__ == "__main__":
    # 制作参数：高斯峰和每个工艺所做的片子
    guass_number = 2
    piece_each_process = 2

    # 输入文件位置
    spectrum_path_origin = 'data/第一轮的数据-PL(统一前标).csv' # 输入的光谱文件数据（原始）
    process_data_path = "data/initial_data_R2.xlsx"  # 输入的工艺参数位置

    # 中间数据的存储位置
    spectrum_path_processed = "各片代表性光谱.csv" # 存各片光谱
    fit_params_path = "拟合参数第一轮.csv" # 光谱拟合参数位置
    input_of_bayesianOpt = "initial_data_with_features.xlsx" # 贝叶斯优化输入

    # 结果存储位置
    final_output_path = "BO_Round1.xlsx"  # 最终保存的输出结果的位置

    # 主函数流程
    # 1.数据处理
    processing_datas = analyze_spectral_uniformity(file_path=spectrum_path_origin,
                                                   piece_each_process=piece_each_process,
                                                   show_plots=True,
                                                   save_results=True,
                                                   save_path=spectrum_path_processed
                                                   )
    # 2.拟合光谱数据
    finial_params = guassFitData(guass_number,
                                 spectrum_path_processed,
                                 set_bounds="default",
                                 save_params_path = fit_params_path,
                                 save_and_view_training_path = None
                                 )
    # 3.提取光谱数据，结合工艺参数，得到贝叶斯优化输入
    df = extract_features_and_merge(fit_params_path=fit_params_path,
                                    process_data_path=process_data_path,
                                    output_path=input_of_bayesianOpt
                                    )
    # 4.贝叶斯优化，输出最后结果
    bayesian_optimization_and_suggest(filepath=input_of_bayesianOpt,
                                    savepath=final_output_path,
                                    target_params="最大峰高",
                                    kind='ucb',
                                    kappa=10,
                                    n_points=25,
                                    )
