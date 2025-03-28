import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# 封装函数
def guassFitData(guass_number,
                 file_path,
                 set_bounds="default",
                 save_params_path = "拟合参数.csv",
                 save_and_view_training_path = None,
                 print_training_params = False
                 ):
    # 函数参数说明：
    # guass_number: 高斯函数峰的个数，可以指定用多少个高斯函数峰去拟合整个曲线
    # file_path: 输入的文件路径，需要使用csv格式的文件
    # bounds: 参数的范围，可以自定义，
            # 默认为bounds自动生成
            # 自定义输入应该按照参考格式来输入， 顺序是（[均值]，[标准差]，[系数]）* 高斯函数峰数量
    # save_params_path: 保存数据的文件名，默认保存到同一文件夹下的"拟合参数.csv"文件中，需要保存到csv文件
    # save_and_view_training_path：训练过程数据可视化，输入为需要保存数据的目标文件夹名称，不存在的话会自动创建。
            # 训练过程的数据包括如下部分：
            # 1.每个光谱图的图像以及拟合函数曲线, 保存到此输入的文件夹中
            # 2.每个光谱图像的R^2, 保存到此输入的文件夹中的folder_path+"R2.csv"文件中
    # 返回值：为pd.DataFrame类型数据，保存所有的参数

    def gaussian_sum(x, *params):
        # params应该是guass_number组（均值，标准差，系数）
        assert len(params) == 3*guass_number  # guass_number个正态分布，每个正态分布3个参数
        y = np.zeros_like(x, dtype=np.float64)
        for i in range(guass_number):
            mean = params[3 * i]
            stddev = params[3 * i + 1]
            amplitude = params[3 * i + 2]
            y += amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
        return y

    def exp(x, mean, stddev, amplitude):
        y = amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
        return y

    def plot_exp(x, mean, stddev, amplitude):
        y = exp(x, mean, stddev, amplitude)
        plt.plot(y,x)

    def R_square(actually_data,predict_data):
        actually_data = np.array(actually_data)
        predict_data = np.array(predict_data)
        RSS = np.sum(np.square(actually_data-predict_data))
        TSS = np.sum(np.square(actually_data-np.mean(predict_data)))
        R2 = 1-RSS/TSS
        return R2

    if file_path[-3:] != "csv":
        # 要求文件必须为csv格式，保证读取稳定性
        # 也可以加一个处理代码：判断文件类型，然后用openxyl或者pd读取，自动另存为csv也可
        raise TypeError("文件必须为csv格式，可以用excel将文件另存为csv格式后重新输入")


    f = open(file_path,"r", encoding = 'utf-8')
    xdata = []
    ydata = []
    ytitle = []
    R2_list = []
    title = True

    # for i in f.readlines():
    #     if title:
    #         i = i.replace('\n','')
    #         i = i.split(',')
    #         for j in i[2:]:
    #             xdata.append(eval(j))
    #         title=False
    #     else:
    #         i = i.replace('\n', '')
    #         i = i.split(',')
    #         ytitle.append(i[0]+'-'+i[1])
    #         for j in i[2:]:
    #             ydata.append(eval(j))

    for i in f.readlines():
        if title:
            i = i.replace('\n','')
            i = i.split(',')
            for j in i[1:]:
                xdata.append(eval(j))
            title=False
        else:
            i = i.replace('\n', '')
            i = i.split(',')
            # ytitle.append(i[0]+'-'+i[1])
            ytitle.append(i[0])
            for j in i[1:]:
                ydata.append(eval(j))

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ytitle = np.array(ytitle)
    ydata = ydata.reshape(ydata.shape[0]//xdata.shape[0], xdata.shape[0])
    xdata = xdata.T
    ydata = ydata.T
    ytitle = ytitle.reshape(ytitle.shape[0], 1)

    def get_params(y):
        initial_params = np.zeros((3*guass_number))
        for i in range(guass_number): # 随机数调整
            initial_params[i * 3] = min(xdata)+(i)/guass_number*(max(xdata)-min(xdata))
            initial_params[i * 3 + 1] = np.random.rand() * 200
            initial_params[i * 3 + 2] = np.random.rand() * 0.5 * max(y)
        return initial_params

    def get_bounds(x, y):
        # 设置参数界限：
        # 简单版：取0~正无穷
        # bounds = ([0 for i in range(guass_number * 3)],[np.inf for i in range(guass_number * 3)])

        # 强化版：
        # 均值 ∈ [光谱波长x最小值， 光谱波长x最大值]
        # 标准差 ∈ [1, 500] (避免标准差为0，设置最小值为1)
        # 系数 ∈ [0， 2*光谱波长y最大值] (允许峰高超过当前最大值)
        lower_bounds = []
        upper_bounds = []
        for i in range(guass_number):
            lower_bounds.extend([min(x), 1, 0])  # 均值, 标准差, 系数
            upper_bounds.extend([max(x), 500, max(y)])
        return (lower_bounds, upper_bounds)



    finial_params = np.array([])
    y_fits = []

    # 拟合数据

    for index in range(ydata.shape[1]):
        R2 = 0
        while R2<0.8:
            # 初始参数随机生成
            initial_params = get_params(ydata[:,index])

            if set_bounds == "default":
                # 如果选择默认参数范围（均大于0）
                # 直接随机生成
                bounds = get_bounds(xdata, ydata[:,index])
            else:
                bounds = set_bounds

            try:
                popt, pcov = curve_fit(gaussian_sum, xdata, ydata[:,index], p0=initial_params, maxfev=2000, bounds = bounds)
            except RuntimeError:
                R2 = 0
                continue
            except ValueError:
                print(initial_params)
                print(bounds)

            # 预测
            y_fit = gaussian_sum(xdata, *popt)

            #计算并保存效果
            R2 = R_square(ydata[:, index], y_fit)
            R2_list.append(R2)

            # 输出拟合参数（如果需要）
            if print_training_params:
                print("Fitted parameters of {}:".format(ytitle[index][0]))
                for i in range(guass_number):
                    print(f"Gaussian {i+1}: Mean = {popt[3*i]}, Stddev = {popt[3*i+1]}, Amplitude = {popt[3*i+2]}")

                print("R^2 of {}: {}".format(ytitle[index][0], R2))

            # 作图
            if save_and_view_training_path == None:
                pass
            else:
                # 需要输出和观察训练过程
                folder_path = save_and_view_training_path
                if not os.path.exists(folder_path):
                    # 无则创建
                    os.makedirs(folder_path)
                # 绘制结果
                plt.figure(figsize=(10, 5))
                plt.scatter(xdata, ydata[:, index], color='r', label='Observations', s=10)
                plt.plot(xdata, y_fit, color='b', label='Fitted Gaussian Sum', linewidth=2)
                for i in range(guass_number):
                    plt.plot(xdata, exp(xdata, popt[3 * i + 0], popt[3 * i + 1], popt[3 * i + 2]), "--")
                plt.title('Least Squares Fit of Gaussian Sum of Process No.' + ytitle[index][0])
                plt.legend()
                # fig_name = ytitle[index][0] + ".svg"  # str
                fig_name = str(index) + ".svg"
                plt.savefig(os.path.join(folder_path, fig_name), dpi=300)
                # plt.show()
                plt.close()

        # 记录参数
        finial_params = np.append(finial_params, popt)


    # 保存参数
    finial_params = finial_params.reshape(len(finial_params)//(guass_number * 3), guass_number * 3)
    # finial_params = np.concatenate((ytitle, finial_params), axis = 1)
    columns_name = ["均值1", "标准差1", "峰高1", "均值2", "标准差2", "峰高2"]
    finial_params = pd.DataFrame(finial_params, columns=columns_name, index = ytitle.reshape(-1))



    if save_params_path[-3:] != "csv":
        raise TypeError("应该保存到csv文件中")
    else:
        finial_params.to_csv(save_params_path, encoding='utf-8')

    if save_and_view_training_path != None:
        R2_path = folder_path+"R2.csv"
        finial_params.to_csv(R2_path, encoding='utf-8_sig')

    return finial_params

# 示例g
if __name__ == "__main__":
    # guassFitData(2, '第一轮数据-PL-处理成完.csv', save_params_path="拟合参数try.csv")
    # guassFitData(2, "第二轮的数据-PL(修正后).csv", save_params_path="拟合参数第二轮.csv", save_and_view_training_path="image")
    guassFitData(2,
                 "代表性光谱.csv",
                 save_params_path="拟合参数第一轮.csv",
                 save_and_view_training_path="image")