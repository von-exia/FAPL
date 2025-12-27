from __future__ import print_function
import MNN.numpy as np
import MNN
import MNN.cv as cv2
import MNN.nn as nn
from time import time
import os
import glob
import numpy as np_std
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm

def inference_single(net, img_path):
    """ inference ViT using a single picture """
    # 预处理
    image = cv2.imread(img_path)
    # cv2 read as bgr format
    image = image[..., ::-1]  # change to rgb format
    image = cv2.resize(image, (224, 224)) / 255.
    # resize to mobile_net tensor size
    image = image - (0.48145466, 0.4578275, 0.40821073)
    image = image / (0.26862954, 0.26130258, 0.27577711)
    # Make var to save numpy; [h, w, c] -> [n, h, w, c]
    input_var = np.expand_dims(image, [0])
    # cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    
    # inference
    output_var = net.forward([input_var])
    predict = output_var[0][0][0]  # 假设输出是单个概率值
    return float(predict)

def load_dataset_from_txt(txt_path, root="../df_datasets/VideoCDF", num_images_per_dir=32):
    """从txt文件加载数据集，每个目录取前num_images_per_dir张图像"""
    dataset = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        line = line.replace('/home/liu/fcb/dataset/VideoDFDCP', root)
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            continue
            
        dir_path, label = parts
        if isinstance(label, str):
            if label == "True":
                label = 1
            else:
                label = 0
        label = int(label)
        
        # 获取目录下所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(dir_path, ext)
            image_files.extend(glob.glob(pattern))
        
        # 按文件名排序并取前num_images_per_dir张
        image_files.sort()
        image_files = image_files[:num_images_per_dir]
        
        # 添加到数据集
        for img_path in image_files:
            dataset.append({
                'path': img_path,
                'label': label,
                'dir': dir_path
            })
    return dataset

def evaluate_dataset(net, dataset):
    """评估整个数据集（使用tqdm包装）"""
    all_predictions = []
    all_labels = []
    inference_times = []
    
    print(f"开始评估，总共 {len(dataset)} 张图像...")
    
    # 使用tqdm包装推理过程
    with tqdm(total=len(dataset), desc="推理进度", unit="张", ncols=100) as pbar:
        for i, data in enumerate(dataset):
            img_path = data['path']
            label = data['label']
            
            # 推理单张图像
            st = time()
            try:
                prediction = inference_single(net, img_path)
                ed = time()
                
                all_predictions.append(prediction)
                all_labels.append(label)
                inference_times.append(ed - st)
                
                # 更新进度条描述
                pbar.set_postfix({
                    "预测值": f"{prediction:.4f}",
                    "标签": label,
                    "平均时间": f"{sum(inference_times)/len(inference_times):.3f}s"
                })
                pbar.update(1)
                
            except Exception as e:
                print(f"\n处理图像 {img_path} 时出错: {e}")
                pbar.update(1)
                continue
    
    # 计算统计信息
    if len(inference_times) > 0:
        avg_time = sum(inference_times) / len(inference_times)
        print(f"\n推理统计:")
        print(f"  平均推理时间: {avg_time:.4f} 秒/张")
        print(f"  总推理时间: {sum(inference_times):.2f} 秒")
        print(f"  处理图像数: {len(all_predictions)}")
    
    return all_predictions, all_labels

def calculate_metrics(y_true, y_pred):
    """计算AUC、AP、EER等指标"""
    # 转换为numpy数组
    y_true = np_std.array(y_true)
    y_pred = np_std.array(y_pred)
    
    if len(set(y_true)) < 2:
        print("警告: 标签只有一种类别，无法计算AUC、AP、EER")
        return {
            'auc': 0.0,
            'ap': 0.0,
            'eer': 0.0,
            'fpr': np_std.array([]),
            'tpr': np_std.array([]),
            'precision': np_std.array([]),
            'recall': np_std.array([]),
            'thresholds': np_std.array([])
        }
    
    # 计算AUC
    auc_score = roc_auc_score(y_true, y_pred)
    
    # 计算AP (Average Precision)
    ap_score = average_precision_score(y_true, y_pred)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    # 计算EER (Equal Error Rate)
    eer = calculate_eer(fpr, tpr)
    
    # 计算Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    return {
        'auc': auc_score,
        'ap': ap_score,
        'eer': eer,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }

def calculate_eer(fpr, tpr):
    """计算EER (Equal Error Rate)"""
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def plot_curves(metrics, save_dir='.'):
    """绘制ROC曲线和PR曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制ROC曲线
    ax1.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {metrics["auc"]:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 标注EER点
    eer = metrics['eer']
    eer_idx = np_std.argmin(np_std.abs(metrics['fpr'] - eer))
    eer_tpr = metrics['tpr'][eer_idx]
    ax1.plot(eer, eer_tpr, 'ro', markersize=8, label=f'EER = {eer:.4f}')
    ax1.plot([eer, eer], [0, eer_tpr], 'r--', lw=1)
    ax1.plot([0, eer], [eer_tpr, eer_tpr], 'r--', lw=1)
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('假阳率 (FPR)')
    ax1.set_ylabel('真阳率 (TPR)')
    ax1.set_title('ROC曲线')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 绘制PR曲线
    ax2.plot(metrics['recall'], metrics['precision'], color='darkgreen', lw=2,
             label=f'PR曲线 (AP = {metrics["ap"]:.4f})')
    
    # 随机分类器的基线
    positive_ratio = sum(y_true) / len(y_true) if 'y_true' in locals() else 0.5
    ax2.plot([0, 1], [positive_ratio, positive_ratio], color='navy', 
             lw=2, linestyle='--', label=f'随机分类器 (AP = {positive_ratio:.4f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('召回率 (Recall)')
    ax2.set_ylabel('精确率 (Precision)')
    ax2.set_title('Precision-Recall曲线')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    roc_path = os.path.join(save_dir, 'roc_pr_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"ROC和PR曲线已保存为 '{roc_path}'")
    
    # 显示图像
    plt.show()
    
    return fig

def print_detailed_results(y_true, y_pred, dataset, metrics):
    """打印详细结果"""
    # 按目录分组统计
    dir_stats = {}
    
    for i, data in enumerate(dataset):
        if i >= len(y_pred):
            break
            
        dir_path = data['dir']
        if dir_path not in dir_stats:
            dir_stats[dir_path] = {
                'predictions': [],
                'labels': [],
                'count': 0
            }
        
        dir_stats[dir_path]['predictions'].append(y_pred[i])
        dir_stats[dir_path]['labels'].append(y_true[i])
        dir_stats[dir_path]['count'] += 1
    
    print("\n" + "="*80)
    print("模型性能评估结果")
    print("="*80)
    
    print(f"\n总体指标:")
    print(f"  AUC (ROC曲线下面积): {metrics['auc']:.4f}")
    print(f"  AP  (平均精确率):     {metrics['ap']:.4f}")
    print(f"  EER (等错误率):      {metrics['eer']:.4f}")
    
    # 找到最佳阈值（Youden's J statistic）
    youden_j = metrics['tpr'] - metrics['fpr']
    best_idx = youden_j.argmax()
    best_threshold = metrics['thresholds'][best_idx]
    print(f"  最佳阈值 (Youden's J): {best_threshold:.4f}")
    print(f"  对应TPR: {metrics['tpr'][best_idx]:.4f}, FPR: {metrics['fpr'][best_idx]:.4f}")
    
    print(f"\n正负样本分布:")
    positive_count = sum(y_true)
    negative_count = len(y_true) - positive_count
    print(f"  正样本 (1): {positive_count} 张")
    print(f"  负样本 (0): {negative_count} 张")
    print(f"  正样本比例: {positive_count/len(y_true):.2%}")
    
    print("\n按目录统计结果:")
    print("-" * 80)
    for dir_path, stats in dir_stats.items():
        avg_pred = np_std.mean(stats['predictions'])
        true_label = stats['labels'][0]  # 同一目录下所有图像标签相同
        print(f"目录: {dir_path}")
        print(f"  标签: {true_label}")
        print(f"  图像数量: {stats['count']}")
        print(f"  平均预测值: {avg_pred:.4f}")
        print(f"  预测值范围: [{min(stats['predictions']):.4f}, {max(stats['predictions']):.4f}]")
        print()

def save_results_to_csv(y_true, y_pred, dataset, metrics, output_dir='.'):
    """保存结果到CSV文件"""
    import csv
    import pandas as pd
    
    # 保存详细推理结果
    detailed_path = os.path.join(output_dir, 'inference_results.csv')
    with open(detailed_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "predicted_probability", "directory"])
        for i, data in enumerate(dataset):
            if i < len(y_pred):
                writer.writerow([data['path'], data['label'], f"{y_pred[i]:.6f}", data['dir']])
    
    print(f"详细推理结果已保存到 '{detailed_path}'")
    
    # 保存指标摘要
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    summary_data = {
        'Metric': ['AUC', 'AP', 'EER', '样本总数', '正样本数', '负样本数'],
        'Value': [
            f"{metrics['auc']:.4f}",
            f"{metrics['ap']:.4f}",
            f"{metrics['eer']:.4f}",
            len(y_true),
            sum(y_true),
            len(y_true) - sum(y_true)
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"指标摘要已保存到 '{summary_path}'")

def main():
    # 模型加载配置
    config = {
        'precision': 'low',  # 当硬件支持（armv8.2）时使用fp16推理
        'backend': 0,       # CPU
        'numThread': 4      # 线程数
    }
    
    # 创建运行时管理器
    rt = nn.create_runtime_manager((config,))
    
    # 加载模型
    model_path = "./mnn_model./FAPL_detector.mnn"
    print(f"正在加载模型: {model_path}")
    net = MNN.nn.load_module_from_file(
        model_path, 
        ["img"], 
        ["prob"],
        runtime_manager=rt
    )
    print("模型加载完成!\n")
    
    # 加载数据集
    root = "../df_datasets/VideoDFDCP"
    # root = "../df_datasets/VideoCDF"
    txt_path = root + "/all.txt"  
    num_images_per_dir = 32
    
    print(f"从 {txt_path} 加载数据集...")
    dataset = load_dataset_from_txt(txt_path, root=root, num_images_per_dir=num_images_per_dir)
    print(f"数据集加载完成，共 {len(dataset)} 张图像\n")
    
    # 评估数据集
    y_pred, y_true = evaluate_dataset(net, dataset)
    
    if len(y_pred) == 0 or len(y_true) == 0:
        print("错误: 没有成功处理任何图像")
        return
    
    # 计算所有指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 打印详细结果
    print_detailed_results(y_true, y_pred, dataset, metrics)
    
    # 绘制曲线
    plot_curves(metrics)
    
    # 保存结果
    save_results_to_csv(y_true, y_pred, dataset, metrics)
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()