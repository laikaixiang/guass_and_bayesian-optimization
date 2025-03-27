from data_processing import analyze_spectral_uniformity
from BO_function import bayesian_optimization_and_suggest
from guass_function import guassFitData

if __name__ == "__main__":
    guass_number = 2
    file_path = "代表性光谱.csv"
    savepath = "BO_Round1.csv"
    processing_datas = analyze_spectral_uniformity(
        'data/第一轮的数据-PL(统一前标).csv',
        piece_each_process=2
    )
    guassFitData(guass_number,
                 file_path,
                 bounds="default",
                 save_params_path = "拟合参数.csv",
                 save_and_view_training_path = None
                 )
    bayesian_optimization_and_suggest("拟合参数.csv",
                                      savepath,
                                      kind='ucb',
                                      kappa=10,
                                      n_points=5)
