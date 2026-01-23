import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')  # 使用非交互式后端
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_minco_result(file_path):
    """
    加载MINCO优化结果

    参数:
    file_path: npz文件路径

    返回:
    包含优化结果的字典
    """
    try:
        data = np.load(file_path)
        return {
            'grid': data['grid'],
            'resolution': data['resolution'],
            'origin': data['origin'],
            'oc_list': data['oc_list'],
            'r_list': data['r_list'],
            'path': data['path'],
            'traj': data['traj'],
            'q_opt': data['q_opt'],
            'T_opt': data['T_opt']
        }
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

def plot_segment_time_histogram(data, save_path=None):
    """
    绘制多项式段时长的直方图

    参数:
    data: MINCO结果数据字典
    save_path: 保存路径，如果为None则显示图像
    """
    if data is None or 'T_opt' not in data:
        print("数据无效或缺少T_opt")
        return

    T_opt = data['T_opt']

    # 使用原始的T_opt顺序，不进行排序
    segment_times = T_opt

    # 创建直方图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制直方图
    bars = ax.bar(range(len(segment_times)), segment_times,
                  color='skyblue', edgecolor='navy', alpha=0.7, width=0.6)

    # 添加数值标签
    for i, (bar, time) in enumerate(zip(bars, segment_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(segment_times)*0.01,
                f'{time:.3f}', ha='center', va='bottom', fontsize=10)

    # 设置标题和标签
    ax.set_title('MINCO Trajectory Optimization - Polynomial Segment Time Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Segment Index', fontsize=14)
    ax.set_ylabel('Duration (seconds)', fontsize=14)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 设置x轴刻度
    ax.set_xticks(range(len(segment_times)))
    ax.set_xticklabels([f'Seg{i+1}' for i in range(len(segment_times))], rotation=45)

    # 添加统计信息
    total_time = np.sum(segment_times)
    mean_time = np.mean(segment_times)
    max_time = np.max(segment_times)
    min_time = np.min(segment_times)

    stats_text = f'Total Time: {total_time:.3f}s\nMean Time: {mean_time:.3f}s\nMax Time: {max_time:.3f}s\nMin Time: {min_time:.3f}s'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 加载MINCO结果
    data_file = os.path.join(script_dir, "minco_result.npz")
    data = load_minco_result(data_file)

    if data:
        # 绘制直方图，保存到脚本所在目录
        save_path = os.path.join(script_dir, "segment_time_histogram.png")
        plot_segment_time_histogram(data, save_path=save_path)
    else:
        print("无法加载MINCO结果数据")