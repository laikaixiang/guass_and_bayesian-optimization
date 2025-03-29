import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def analyze_spectral_uniformity(file_path,
                                piece_each_process: int,
                                show_plots=True,
                                save_results=True,
                                save_path="各片代表性光谱"):
    """
    分析光谱数据均匀性并识别离群值的完整函数

    参数:
        file_path (str): CSV文件路径
        piece_each_process (int): 一个工艺制作了多少个片子:1/2(不会有人要做更多了吧TAT)
        show_plots (bool): 是否显示分析图表(默认True)
        save_results (bool): 是否保存结果文件(默认True)

    返回:
        dict: 包含分析结果的字典，包括:
            - 'cleaned_data': 处理后的光谱数据
            - 'representative_spectrum': 代表性光谱
            - 'uniformity_scores': 每组均匀性分数
            - 'avg_uniformity': 平均均匀性
            - 'cv_by_wavelength': 各波长的变异系数

    文件输出：
        representative_spectrum.csv: 全部组的代表性光谱数据
        spectral_analysis_result: 结果图（图一、图二为：以第一组数据为例的数据清洗示例；图三：所有组的均匀性分数
        uniformity_scores: 均匀性评分
    """

    # 1. 数据读取与预处理
    def load_and_preprocess_data(file_path):
        """
        加载并预处理光谱数据
        wavelengths：光的波长
        sample_data：原本的强度数据
        grouped_data：分组（4个为一组）的数据
        """
        data = pd.read_csv(file_path, index_col=0)
        wavelengths = data.columns.astype(int)
        sample_data = data.values
        grouped_data = np.array(np.split(sample_data, len(sample_data) // 4))
        index_data = np.array(data.index[i*4] for i in range(len(sample_data) // 4))
        return wavelengths, sample_data, grouped_data, index_data


    # 2. 离群值检测与处理
    def detect_and_clean_outliers(grouped_data: np.ndarray):
        """检测并处理离群值"""

        def process_group(group):
            median = np.median(group, axis=0) # 中位数
            q1 = np.percentile(group, 25, axis=0) # 25%位数
            q3 = np.percentile(group, 75, axis=0) # 75%位数
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            cleaned_group = group.copy() # 深拷贝
            for i in range(group.shape[0]):
                outlier_mask = (group[i] < lower_bound) | (group[i] > upper_bound)
                cleaned_group[i, outlier_mask] = median[outlier_mask]

            return cleaned_group

        return np.vstack([process_group(group) for group in grouped_data])

    # 3. 均匀性评估
    def calculate_uniformity_metrics(cleaned_data, grouped_data_shape):
        """计算均匀性指标
            使用变异系数进行计算：计算四个光谱的离散程度
        """
        # 重新分组处理后的数据
        cleaned_groups = np.array(np.split(cleaned_data, len(cleaned_data) // 4))

        # 计算每组均匀性
        def group_uniformity(group):
            # 我们用变异系数来评价四个点的离散程度，然后通过一个组的变异系数来确定每一条光谱线的平均异变系数
            # 这是为了评价这个片子的均匀性，越均匀=>离散程度越低
            std_dev = np.std(group, axis=0) # 计算每一列（4个）的标准差
            mean_val = np.mean(group, axis=0) # 计算每一列的均值
            cv = np.where(mean_val != 0, std_dev / mean_val, 0)
            return 1 - np.mean(cv)

        uniformities = [group_uniformity(group) for group in cleaned_groups] # cleaned gounp size:40*4*101

        # 计算代表性光谱(各组中位数的中位数)
        representative_spectra = np.median(cleaned_groups, axis=1)
        # final_representative = np.median(representative_spectra, axis=0)

        # 计算各波长的变异系数
        cv_by_wavelength = np.std(cleaned_data, axis=0) / np.mean(cleaned_data, axis=0)

        return {
            'uniformity_scores': uniformities,
            'avg_uniformity': np.mean(uniformities),
            'representative_spectrum': representative_spectra, # 每组中位数
            'cv_by_wavelength': cv_by_wavelength
        }

    # 4、工艺参数估算
    def evaluate_process_quality(metrics):
        uniformity_scores = metrics["uniformity_scores"]
        data_processes = np.split(metrics['representative_spectrum'],
                                  len(metrics['representative_spectrum']) // piece_each_process) # 按照一个工艺两个片子来分数据
        index_processes = np.array([i//piece_each_process for i in range(len(uniformity_scores))])
        delete_index = [] # 指定删除的节点
        length_of_ws = len(data_processes[0][0])
        for process in range(len(data_processes)):
            # 如果缺数据，标0，那就去掉
            if piece_each_process == 2:
                # 只有两个片子的用到这个：
                frist_piece_nan = (data_processes[process][0][0] == 0) # 第一个片子没测到
                second_piece_nan = (data_processes[process][1][0] == 0) # 第二个片子没测到
                frist_piece_nonuniform = (uniformity_scores[process * 2] < 0.7) # 第一个片子不均匀
                second_piece_nonuniform = (uniformity_scores[process * 2 + 1] < 0.7)  # 第二个片子不均匀
                if frist_piece_nan or second_piece_nan:
                    # （一组中）有片子不均匀
                    if frist_piece_nan and second_piece_nan:
                        # 直接全删
                        data_processes[process] = np.array([[]])
                        delete_index.append(process * 2)
                        delete_index.append(process * 2 + 1)
                    elif frist_piece_nan:
                    # if data_processes[process][0][0] == 0:
                        data_processes[process] = data_processes[process][1].reshape((1, length_of_ws))
                        delete_index.append(process * 2)
                    # elif data_processes[process][1][0]==0:
                    elif second_piece_nan:
                        data_processes[process] = data_processes[process][0].reshape((1, length_of_ws))
                        delete_index.append(process * 2 + 1)

                # 不够均匀的也删掉
                # if uniformity_scores[process * 2] < 0.7 or uniformity_scores[process * 2] < 0.7:
                if frist_piece_nonuniform or second_piece_nonuniform:
                    # 先讨论两个都是不均匀的
                    if frist_piece_nonuniform and second_piece_nonuniform:
                        # 用均值代替，删去后面那个值
                        data_processes[process] = (np.mean(
                                                [data_processes[process][0], data_processes[process][1]],
                                                    axis=0)
                                                    .reshape(1, length_of_ws))
                        delete_index.append(process * 2)
                    elif frist_piece_nonuniform:
                    # elif uniformity_scores[process * 2] < 0.7:
                        data_processes[process] = data_processes[process][1].reshape((1, length_of_ws))
                        delete_index.append(process * 2)
                    elif second_piece_nonuniform:
                    # elif uniformity_scores[process * 2 + 1] < 0.7:
                        # data_processes[process][1]==0
                        data_processes[process] = data_processes[process][0].reshape((1, length_of_ws))
                        delete_index.append(process * 2 + 1)
            else:
                # 一个片子，如果不够均匀就直接删去
                if uniformity_scores[process] < 0.7:
                    delete_index.append(process)

        if piece_each_process == 1:
            # 只做了一个片子的时候要单独删
            data_processes = np.delete(np.array(data_processes), delete_index,axis=0)

        # 剩下的做一个片子和两个片子是一样的
        uniformity_scores_cleaned = np.delete(uniformity_scores, delete_index)
        index_processes = np.delete(index_processes, delete_index)
        return {
            'processes_data': data_processes, # detype: list
            'cleaned_uniformity': np.array(uniformity_scores_cleaned),
            'index_processes': np.array(index_processes) # 指示，和cleaned_uniformity配合着用，可以求出每个片子的平均均匀度
        }

    # 5. 可视化
    def create_visualizations(wavelengths, original_data, cleaned_data, metrics, processes):
        """创建分析图表"""
        plt.figure(figsize=(18, 12))
        # 调整子图位置
        plt.subplots_adjust(
            left=0.062,  # 左边距（默认0.125）
            right=0.985,  # 右边距
            bottom=0.07,  # 底边距（默认0.11）
            top=0.959,  # 顶边距（默认0.88）
            wspace=0.168,  # 水平间距（默认0.2）
            hspace=0.252  # 垂直间距（默认0.2）
        )

        # 图1: 原始数据与处理后数据对比(第一组)
        plt.subplot(2, 2, 1)
        for i in range(4):
            plt.plot(wavelengths, original_data[i], alpha=0.5, label=f'Original {i + 1}')
        for i in range(4):
            plt.plot(wavelengths, cleaned_data[i], '--', alpha=0.8, label=f'Cleaned {i + 1}')
        plt.title('Original vs Cleaned Spectra (Example: First Group)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)


        # 图2：展示误差（以图一为例)，25%和75%分位数
        plt.subplot(2, 2, 2)
        # 原始数据 - 分位数填充
        q1_original = np.percentile(original_data[:4], 25, axis=0)
        q3_original = np.percentile(original_data[:4], 75, axis=0)
        plt.fill_between(wavelengths, q1_original, q3_original,
                         color='blue', alpha=0.15, label='Original IQR')
        # 清洗数据 - 误差棒
        mean_cleaned = np.mean(cleaned_data[:4], axis=0)
        std_cleaned = np.std(cleaned_data[:4], axis=0)
        plt.errorbar(wavelengths[::10], mean_cleaned[::10], yerr=std_cleaned[::10],
                     fmt='ro', markersize=4, capsize=3, label='Cleaned Mean±Std')
        # 叠加中位数和均值线（修正此处错误）
        plt.plot(wavelengths, np.median(original_data[:4], axis=0), 'b-',
                 linewidth=1, alpha=0.7, label='Original Median')  # 正确调用np.median
        plt.plot(wavelengths, mean_cleaned, 'r--',
                 linewidth=1, label='Cleaned Mean')
        plt.title('Hybrid Visualization: IQR vs Error Bars')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)


        # 图3: 均匀性指标分布
        plt.subplot(2, 2, 3)
        # 设置颜色：>=0.7 为可取（蓝绿色），<0.7 为不可取（橙红色）
        colors = ['#3498db' if score >= 0.7 else '#e74c3c' for score in metrics['uniformity_scores']]
        plt.bar(range(1, len(metrics['uniformity_scores']) + 1),
                [metrics['uniformity_scores'][idx] if metrics['uniformity_scores'][idx]!=1
                 else 0
                for idx in range(len(metrics['uniformity_scores']))
                ],
                color=colors)
        plt.axhline(y=metrics['avg_uniformity'], color='r', linestyle='--',
                    label=f'Average: {metrics["avg_uniformity"]:.3f}')
        plt.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        plt.axhspan(0.7, 1.0, alpha=0.1, hatch='//')
        plt.title('Uniformity Index for Each Sample Group')
        plt.xlabel('Sample Group')
        plt.ylabel('Uniformity Index (0-1)')
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        plt.grid(True)

        # 图4: 所有光谱集合
        plt.subplot(2, 2, 4)
        peak_intensity = [] # 每种制作方法的强度最大值列表
        processes_data = processes["processes_data"]
        index_process = processes["index_processes"]
        uniformity_process = processes["cleaned_uniformity"]
        avg_cleaned_uniformity = [] # 每种制作方法的均匀性均值
        for process in range(len(processes_data)):
            peak_intensity.append(np.max(processes_data[process]))
            avg_cleaned_uniformity.append(
                np.mean(uniformity_process[np.where(index_process == process)]))
        x = np.arange(len(peak_intensity))

        # 设置颜色映射（从红到绿表示综合质量越来越好）
        avg_cleaned_uniformity = np.array(avg_cleaned_uniformity)
        peak_intensity = np.array(peak_intensity)
        
        mask = avg_cleaned_uniformity > 0.4  # 过滤掉极低强度点
        filtered_uniformity = np.array(avg_cleaned_uniformity)[mask]
        filtered_intensity = np.array(peak_intensity)[mask]

        # 动态计算点大小范围（排除最低值）
        min_size = 50  # 最小点大小
        max_size = 500  # 最大点大小
        sizes = np.interp(filtered_intensity,
                          (np.min(filtered_intensity), np.max(filtered_intensity)),
                          (min_size, max_size))

        # 颜色映射（从红到绿）
        quality_score = filtered_uniformity * filtered_intensity / np.max(filtered_intensity)
        cmap = plt.cm.get_cmap('RdYlGn')

        # 绘制散点图
        sc = plt.scatter(
            x=filtered_uniformity,
            y=filtered_intensity,
            c=quality_score,
            cmap=cmap,
            s=sizes,  # 使用动态计算的大小
            alpha=0.8,
            edgecolors='k',  # 黑色边框
            linewidths=0.8,
            zorder=3
        )

        # 坐标轴设置（对数坐标）
        plt.yscale('log')
        plt.xlim(0.3, 1.0)
        plt.ylim(np.min(filtered_intensity) * 0.9, np.max(filtered_intensity) * 1.3)

        # 参考线
        plt.axvline(0.7, color='gray', linestyle=':', alpha=0.5)
        # plt.axhline(10000, color='gray', linestyle=':', alpha=0.5)

        # 颜色条
        cbar = plt.colorbar(sc)
        cbar.set_label('Quality Score\n(Uniformity × Normalized Intensity)',
                       rotation=270, labelpad=20)

        # 标注和标题
        plt.title('Spectral Quality Analysis (Filtered)\n[Size ∝ Intensity]', pad=20, fontsize=14)
        plt.xlabel('Average Uniformity Score (0-1)', fontsize=12)
        plt.ylabel('Peak Intensity (log scale)', fontsize=12)

        # 网格和样式优化
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.gca().set_facecolor('#f8f8f8')

        # 重点标注最高质量点
        max_idx = np.argmax(quality_score)
        plt.scatter(filtered_uniformity[max_idx], filtered_intensity[max_idx],
                    s=max_size * 1.2, facecolors='none', edgecolors='gold',
                    linewidths=2, label='Best Quality')
        plt.legend(loc='upper left')


        # # 双y轴棒图
        # ax1 = plt.gca()
        # ax1.bar(x - 0.2, avg_cleaned_uniformity,
        #         width=0.4, color='#3498db', label='Uniformity')
        # ax1.set_ylabel('Average Uniformity', color='#3498db')
        # ax1.tick_params(axis='y', colors='#3498db')
        # ax1.set_ylim(0, 1)
        # ax1.axhline(0.7, color='#3498db', linestyle=':', alpha=0.5)
        #
        # ax2 = ax1.twinx()
        # ax2.bar(x + 0.2, peak_intensity,
        #         width=0.4, color='#e67e22', label='Intensity')
        # ax2.set_ylabel('Median Intensity', color='#e67e22')
        # ax2.tick_params(axis='y', colors='#e67e22')
        #
        # plt.title('Process Performance Comparison\n(Left: Uniformity, Right: Intensity)')
        # plt.xticks(x, x+1, rotation=45)
        # ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')


        # plt.tight_layout()
        if save_results:
            plt.savefig('spectral_analysis_results.svg', dpi=300, format='svg')
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 主执行流程
    wavelengths, original_data, grouped_data, index_data = load_and_preprocess_data(file_path)
    cleaned_data = detect_and_clean_outliers(grouped_data)
    metrics = calculate_uniformity_metrics(cleaned_data, grouped_data.shape)
    processes = evaluate_process_quality(metrics)

    if show_plots:
        create_visualizations(wavelengths, original_data, cleaned_data, metrics, processes)

    # 保存结果
    if save_results:
        # 保存所有已经筛除的光谱
        flattened = np.concatenate([arr.reshape(-1) for arr in processes["processes_data"]])
        indexs = [p + 1 for p in processes["index_processes"]]
        len_wavelength = len(wavelengths)
        representative_spectrum = flattened.reshape(len(flattened) // len_wavelength, len_wavelength)
        result_df = pd.DataFrame(representative_spectrum,
                                 columns=wavelengths,
                                 index=indexs)
        result_df.to_csv(save_path)

        # 保存均匀性分数
        uniformity_df = pd.DataFrame({
            'Group': range(1, len(metrics['uniformity_scores']) + 1),
            'Uniformity_Score': metrics['uniformity_scores']
        })
        uniformity_df.to_csv('uniformity_scores.csv', index=False)

    return {
        'wavelengths': wavelengths,
        'original_data': original_data,
        'cleaned_data': cleaned_data,
        'representative_spectrum': metrics['representative_spectrum'],
        'uniformity_scores': metrics['uniformity_scores'],
        'avg_uniformity': metrics['avg_uniformity'],
        'cv_by_wavelength': metrics['cv_by_wavelength'],
        'processes_data': processes['processes_data'],  # detype: list
        'cleaned_uniformity': processes['cleaned_uniformity'],
        'index_processes': processes['index_processes']  # 指示，和cleaned_uniformity配合着用，可以求出每个片子的平均均匀度
    }


# 使用示例
if __name__ == "__main__":
    results = analyze_spectral_uniformity('data/第一轮的数据-PL(统一前标).csv',piece_each_process=2)

    # 打印关键结果
    print(f"\n分析完成，平均均匀性分数: {results['avg_uniformity']:.3f}")
    result = np.array(results['uniformity_scores'])
    filter = result[result != 1]
    print(
        f"最佳均匀性组: {np.argmax(results['uniformity_scores']) + 1} (分数: {filter.max():.3f})")
    print(
        f"最差均匀性组: {np.argmin(results['uniformity_scores']) + 1} (分数: {filter.min():.3f})")