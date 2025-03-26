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
        return wavelengths, sample_data, grouped_data

    # 2. 离群值检测与处理
    def detect_and_clean_outliers(grouped_data):
        """检测并处理离群值"""

        def process_group(group):
            median = np.median(group, axis=0)
            q1 = np.percentile(group, 25, axis=0)
            q3 = np.percentile(group, 75, axis=0)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            cleaned_group = group.copy()
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
            std_dev = np.std(group, axis=0)
            mean_val = np.mean(group, axis=0)
            cv = np.where(mean_val != 0, std_dev / mean_val, 0)
            return 1 - np.mean(cv)

        uniformities = [group_uniformity(group) for group in cleaned_groups]

        # 计算代表性光谱(各组中位数的中位数)
        representative_spectra = np.median(cleaned_groups, axis=1)
        final_representative = np.median(representative_spectra, axis=0)

        # 计算各波长的变异系数
        cv_by_wavelength = np.std(cleaned_data, axis=0) / np.mean(cleaned_data, axis=0)

        return {
            'uniformity_scores': uniformities,
            'avg_uniformity': np.mean(uniformities),
            'representative_spectrum': final_representative,
            'cv_by_wavelength': cv_by_wavelength
        }

    # 4. 可视化
    def create_visualizations(wavelengths, original_data, cleaned_data, metrics):
        """创建分析图表"""
        plt.figure(figsize=(18, 12))

        # 图1: 原始数据与处理后数据对比(第一组)
        plt.subplot(2, 2, 1)
        for i in range(4):
            plt.plot(wavelengths, original_data[i], alpha=0.5, label=f'Original {i + 1}')
        for i in range(4):
            plt.plot(wavelengths, cleaned_data[i], '--', alpha=0.8, label=f'Cleaned {i + 1}')
        plt.title('Original vs Cleaned Spectra (First Group)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

        # 图2: 均匀性指标分布
        plt.subplot(2, 2, 2)
        plt.bar(range(1, len(metrics['uniformity_scores']) + 1), metrics['uniformity_scores'])
        plt.axhline(y=metrics['avg_uniformity'], color='r', linestyle='--',
                    label=f'Average: {metrics["avg_uniformity"]:.3f}')
        plt.title('Uniformity Index for Each Sample Group')
        plt.xlabel('Sample Group')
        plt.ylabel('Uniformity Index (0-1)')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)

        # 图3: 代表性光谱(前5组)
        plt.subplot(2, 2, 3)
        cleaned_groups = np.array(np.split(cleaned_data, len(cleaned_data) // 4))
        representative_spectra = np.median(cleaned_groups, axis=1)
        for i, spec in enumerate(representative_spectra[:5]):
            plt.plot(wavelengths, spec, label=f'Group {i + 1}')
        plt.title('Representative Spectra for Sample Groups')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)

        # 图4: 波长变异系数
        plt.subplot(2, 2, 4)
        plt.plot(wavelengths, metrics['cv_by_wavelength'])
        plt.title('Coefficient of Variation Across Wavelengths')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Coefficient of Variation')
        plt.grid(True)

        plt.tight_layout()
        if save_results:
            plt.savefig('spectral_analysis_results.png', dpi=300)
        if show_plots:
            plt.show()
        else:
            plt.close()

    # 主执行流程
    wavelengths, original_data, grouped_data = load_and_preprocess_data(file_path)
    cleaned_data = detect_and_clean_outliers(grouped_data)
    metrics = calculate_uniformity_metrics(cleaned_data, grouped_data.shape)

    if show_plots:
        create_visualizations(wavelengths, original_data, cleaned_data, metrics)

    # 保存结果
    if save_results:
        # 保存代表性光谱
        result_df = pd.DataFrame([metrics['representative_spectrum']],
                                 columns=wavelengths,
                                 index=['Representative_Spectrum'])
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
    results = analyze_spectral_uniformity('第二轮的数据-PL(统一前标).csv')

    # 打印关键结果
    print(f"\n分析完成，平均均匀性分数: {results['avg_uniformity']:.3f}")
    print(
        f"最佳均匀性组: {np.argmax(results['uniformity_scores']) + 1} (分数: {np.max(results['uniformity_scores']):.3f})")
    print(
        f"最差均匀性组: {np.argmin(results['uniformity_scores']) + 1} (分数: {np.min(results['uniformity_scores']):.3f})")