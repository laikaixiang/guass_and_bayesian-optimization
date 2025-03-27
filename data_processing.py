import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def analyze_spectral_uniformity(file_path, show_plots=True, save_results=True):
    """
    分析光谱数据均匀性并识别离群值的完整函数

    参数:
        file_path (str): CSV文件路径
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
        uniformity_scores
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
        """计算均匀性指标"""
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

    # 4. 可视化
    def create_visualizations(wavelengths, original_data, cleaned_data, metrics):
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
        plt.bar(range(1, len(metrics['uniformity_scores']) + 1), metrics['uniformity_scores'], color=colors)
        plt.axhline(y=metrics['avg_uniformity'], color='r', linestyle='--',
                    label=f'Average: {metrics["avg_uniformity"]:.3f}')
        plt.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        plt.axhspan(0.7, 1.0, alpha=0.1, hatch='//')
        plt.title('Uniformity Index for Each Sample Group')
        plt.xlabel('Sample Group')
        plt.ylabel('Uniformity Index (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        # 图4: 所有光谱集合
        plt.subplot(2, 2, 4)
        cleaned_groups = np.array(np.split(cleaned_data, len(cleaned_data) // 4))
        representative_spectra = np.median(cleaned_groups, axis=1)
        for i, spec in enumerate(representative_spectra[:]):
            plt.plot(wavelengths, spec, label=f'Group {i + 1}')
        plt.title('Representative Spectra for Sample Groups')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        # plt.legend()
        plt.grid(True)


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

    if show_plots:
        create_visualizations(wavelengths, original_data, cleaned_data, metrics)

    # 保存结果
    if save_results:
        # 保存所有光谱
        result_df = pd.DataFrame(metrics['representative_spectrum'].reshape((47,101)),
                                 columns=wavelengths,
                                 index=index_data)
        result_df.to_csv('representative_spectrum.csv')

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
        'cv_by_wavelength': metrics['cv_by_wavelength']
    }


# 使用示例
if __name__ == "__main__":
    results = analyze_spectral_uniformity('第一轮的数据-PL(统一前标).csv')

    # 打印关键结果
    print(f"\n分析完成，平均均匀性分数: {results['avg_uniformity']:.3f}")
    print(
        f"最佳均匀性组: {np.argmax(results['uniformity_scores']) + 1} (分数: {np.max(results['uniformity_scores']):.3f})")
    print(
        f"最差均匀性组: {np.argmin(results['uniformity_scores']) + 1} (分数: {np.min(results['uniformity_scores']):.3f})")