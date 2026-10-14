# 二分类 第一次策略
import torch
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
import matplotlib.pyplot as plt
from collections import abc
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, roc_auc_score,cohen_kappa_score,f1_score,precision_score,recall_score



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, exp_log_dir, seed_id):
    log_dir = os.path.join(exp_log_dir, "_seed_" + str(seed_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug("=" * 45)
    logger.debug(f'Seed: {seed_id}')
    logger.debug("=" * 45)
    return logger, log_dir


import os
import torch

def save_checkpoint(exp_log_dir, model, dataset, dataset_configs, hparams, stage, models, best_thresholds=None, best_min_window_counts=None):
    """
    保存模型检查点到指定目录。

    参数:
    - exp_log_dir (str): 实验日志目录。
    - model (torch.nn.Module): 模型实例。
    - dataset (str): 数据集名称。
    - configs: 数据集配置。
    - hparams: 超参数。
    - stage (str): 阶段名称，例如 'best' 或 'last'。
    - models (str): 模型名称。
    - best_thresholds (dict): 最佳阈值。
    - best_min_window_counts (dict): 最小窗口计数。
    """
    # 构建保存目录路径
    save_dir = os.path.join(exp_log_dir, stage)

    # 创建目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 构建保存路径
    save_path = os.path.join(save_dir, f"checkpoint_{models}.pt")

    # 创建要保存的字典
    save_dict = {
        'model': model.state_dict(),
        'dataset': dataset,
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        'best_thresholds': best_thresholds,
        'best_min_window_counts': best_min_window_counts,
    }

    # 保存检查点
    torch.save(save_dict, save_path)







from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import os


# ----------------------------
# 帮助函数：固定二分类混淆矩阵取值顺序（标签 1 为正类）
# 返回 TN, FP, FN, TP
# ----------------------------
def _binary_confusion_counts(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    # cm = [[TN, FP],
    #       [FN, TP]]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    return TN, FP, FN, TP


# ==========
# 二分类总体指标（返回全为百分数）
# ==========
def _calc_metrics(pred_labels, true_labels, pred_probs, classes_names):
    """
    返回 tuple（全部为百分数）：
      Accuracy(micro), F1(macro), Precision(macro), Recall(macro),
      G-Mean(macro), AUC(macro), AUCPR(macro),
      Kappa, MCC
    说明：
      - Accuracy 为微平均（与总体准确率等价）
      - 其余指标均为宏平均（对各类 one-vs-rest 计算后取算术平均）
      - AUCPR 使用平均精度 (Average Precision) 作为 PR AUC 的一致估计
    """
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)
    n_classes = len(classes_names)

    # 若 pred_probs 为一维，则视作正类概率（二分类）
    if pred_probs is not None and np.ndim(pred_probs) == 1:
        pred_probs = np.vstack([1 - pred_probs, pred_probs]).T

    # ----------------
    # 1) 基础（Accuracy micro；其余 macro）
    # ----------------
    accuracy = accuracy_score(true_labels, pred_labels) * 100.0
    macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0) * 100.0
    macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0) * 100.0
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0) * 100.0

    # ----------------
    # 2) G-Mean 的宏平均
    #    逐类 one-vs-rest 计算 TPR/TNR，然后取平均
    # ----------------
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))
    total = cm.sum()
    g_means = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - (tp + fn + fp)

        sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        g_means.append(np.sqrt(sens_i * spec_i))

    macro_g_mean = (float(np.mean(g_means)) * 100.0) if len(g_means) > 0 else 0.0

    # ----------------
    # 3) AUC (macro) 与 AUCPR (macro)
    # ----------------
    macro_auc = 0.0
    macro_aucpr = 0.0
    if pred_probs is not None:
        # AUC：逐类一对其余计算（需要每列为该类的得分/概率）
        auc_values = []
        aucpr_values = []
        for i in range(n_classes):
            class_true = (true_labels == i).astype(int)
            try:
                auc = roc_auc_score(class_true, pred_probs[:, i])
            except Exception:
                auc = 0.0
            auc_values.append(auc)

            try:
                ap = average_precision_score(class_true, pred_probs[:, i])
            except Exception:
                ap = 0.0
            aucpr_values.append(ap)

        macro_auc = float(np.mean(auc_values)) * 100.0 if len(auc_values) > 0 else 0.0
        macro_aucpr = float(np.mean(aucpr_values)) * 100.0 if len(aucpr_values) > 0 else 0.0
    else:
        macro_auc = 0.0
        macro_aucpr = 0.0

    # ----------------
    # 4) Kappa / MCC（全局）
    # ----------------
    kappa = cohen_kappa_score(true_labels, pred_labels) * 100.0
    try:
        mcc = matthews_corrcoef(true_labels, pred_labels) * 100.0
    except Exception:
        mcc = 0.0

    return (
        accuracy,         # Accuracy (micro, %)
        macro_f1,         # F1 (macro, %)
        macro_precision,  # Precision (macro, %)
        macro_recall,     # Recall (macro, %)
        macro_g_mean,     # G-Mean (macro, %)
        macro_auc,        # AUC (macro, %)
        macro_aucpr,      # AUCPR (macro, %)
        kappa,            # Kappa (%)
        mcc               # MCC (%)
    )


def aggregate_subject_predictions(pred_labels, pred_probs, true_labels, subject_ids, method='majority_vote', confidence_threshold=0.6):
    """
    将patch级预测聚合为个体级预测（二分类版本）

    Args:
        pred_labels: patch级预测标签 (N,)
        pred_probs: patch级预测概率 (N, n_classes)
        true_labels: patch级真实标签 (N,)
        subject_ids: patch对应的个体ID (N,)
        method: 聚合方法 ('majority_vote', 'prob_avg', 'weighted_vote')
        confidence_threshold: 加权投票置信度阈值

    Returns:
        tuple: (subject_pred_labels, subject_true_labels, subject_pred_probs, subject_ids_unique)
    """
    import numpy as np
    from collections import Counter

    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    subject_ids = np.array(subject_ids)

    # 处理pred_probs为一维的情况（二分类）
    if pred_probs.ndim == 1:
        pred_probs = np.vstack([1 - pred_probs, pred_probs]).T

    unique_subjects = np.unique(subject_ids)
    n_classes = pred_probs.shape[1]

    subject_pred_labels = []
    subject_true_labels = []
    subject_pred_probs = []
    subject_ids_list = []

    for subject_id in unique_subjects:
        # 获取该个体的所有patch
        subject_mask = subject_ids == subject_id
        subj_pred_labels = pred_labels[subject_mask]
        subj_pred_probs = pred_probs[subject_mask]
        subj_true_labels = true_labels[subject_mask]

        # 个体的真实标签（假设同一个体所有patch标签相同）
        subj_true_label = subj_true_labels[0]

        if method == 'majority_vote':
            # 多数投票
            vote_counts = Counter(subj_pred_labels)
            subj_pred_label = vote_counts.most_common(1)[0][0]
            # 概率为投票比例
            subj_pred_prob = np.zeros(n_classes)
            for label, count in vote_counts.items():
                subj_pred_prob[int(label)] = count / len(subj_pred_labels)

        elif method == 'prob_avg':
            # 概率平均
            subj_pred_prob = np.mean(subj_pred_probs, axis=0)
            subj_pred_label = np.argmax(subj_pred_prob)

        elif method == 'weighted_vote':
            # 加权投票：根据预测置信度加权
            weights = np.max(subj_pred_probs, axis=1)  # 每个patch的最大概率作为权重
            # 只使用高置信度的patch
            confident_mask = weights >= confidence_threshold
            if np.sum(confident_mask) > 0:
                weights = weights[confident_mask]
                subj_pred_probs_confident = subj_pred_probs[confident_mask]
                # 加权平均
                subj_pred_prob = np.average(subj_pred_probs_confident, axis=0, weights=weights)
            else:
                # 如果没有高置信度patch，回退到普通平均
                subj_pred_prob = np.mean(subj_pred_probs, axis=0)
            subj_pred_label = np.argmax(subj_pred_prob)

        elif method == 'threshold_strategy':
            # 阈值策略：基于阈值和最小patch数量约束的聚合
            # 从hparams中获取阈值参数（需要预先优化）
            import sys
            frame = sys._getframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'hparams'):
                    hparams = frame.f_locals['self'].hparams
                    break
                frame = frame.f_back

            if 'hparams' in locals() and 'subject_thresholds' in hparams and 'subject_min_window_counts' in hparams:
                thresholds = hparams['subject_thresholds']
                min_window_counts = hparams['subject_min_window_counts']

                # 应用阈值策略
                subj_pred_label, subj_pred_prob = _apply_threshold_strategy(
                    subj_pred_probs, thresholds, min_window_counts
                )
            else:
                # 如果没有阈值参数，回退到概率平均
                subj_pred_prob = np.mean(subj_pred_probs, axis=0)
                subj_pred_label = np.argmax(subj_pred_prob)

        elif method == 'cf_bgg':
            # CF-BGG策略：Counterfactual-Banzhaf-Gear Graph Aggregation
            # 从hparams中获取参数（需要预先优化）
            import sys
            frame = sys._getframe()
            while frame:
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'hparams'):
                    hparams = frame.f_locals['self'].hparams
                    break
                frame = frame.f_back

            # CF-BGG默认参数（二分类）- 优化版
            cf_bgg_params = {
                'K': min(10, max(1, int(0.25 * len(subj_pred_probs)))),  # 每类Top-K（二分类可以稍大）
                'M': 2,  # 多样性覆盖
                'alpha': 0.6,  # LOO vs Banzhaf权重（偏向LOO，计算更快）
                'S': 16,  # 联盟采样数（二分类可以稍大）
                'sigma': 1.0,  # 相似度带宽（二分类用1维差）
                'lambda_': 1.0,  # 一致性放大
                'beta': 0.6,  # 对比抑制（二分类可以稍小）
                'tau_m': 0.20,  # margin阈值
                'tau_kappa': 0.4,  # 一致性阈值
                'tau_f': 0.10,  # 支持率阈值
                'gamma': 1.0  # 温度参数
            }

            # 如果hparams中有优化后的参数，使用它们
            if 'hparams' in locals() and hasattr(hparams, 'get') and hparams.get('cf_bgg_params'):
                cf_bgg_params.update(hparams['cf_bgg_params'])

            # 应用CF-BGG策略（固定随机种子确保结果可重现）
            subj_pred_label, subj_pred_prob = _apply_cf_bgg_strategy(
                subj_pred_probs, random_seed=42, **cf_bgg_params
            )

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        subject_pred_labels.append(int(subj_pred_label))
        subject_true_labels.append(int(subj_true_label))
        subject_pred_probs.append(subj_pred_prob)
        subject_ids_list.append(int(subject_id))

    return (np.array(subject_pred_labels),
            np.array(subject_true_labels),
            np.array(subject_pred_probs),
            np.array(subject_ids_list))


def _apply_threshold_strategy(probabilities, thresholds, min_window_counts):
    """
    应用阈值策略进行patch级到个体级的聚合（二分类版本）

    Args:
        probabilities: patch级预测概率 (num_patches, num_classes)
        thresholds: 每个类别的概率阈值 {0: 0.5, 1: 0.6}
        min_window_counts: 每个类别的最小patch比例要求 {0: 0.3, 1: 0.4}

    Returns:
        final_class: 最终预测类别
        pred_probs: 最终预测概率分布
    """
    import math

    num_patches = probabilities.shape[0]
    num_classes = probabilities.shape[1]

    # 1) 初始化计数和"超过阈值概率之和"
    class_count = {i: 0 for i in thresholds.keys()}
    class_prob_sum = {i: 0.0 for i in thresholds.keys()}

    # 2) 遍历所有patch的预测概率
    for patch_probs in probabilities:
        for class_index, threshold in thresholds.items():
            class_index = int(class_index)
            # 如果当前patch对该类别的概率 > 阈值，则计数加1，并累加其概率
            if patch_probs[class_index] > threshold:
                class_count[class_index] += 1
                class_prob_sum[class_index] += patch_probs[class_index]

    # 3) 计算每个类别的最小patch数要求（基于比例）
    min_patch_required = {}
    for class_index, proportion in min_window_counts.items():
        # 计算最小patch数，使用ceil确保至少满足比例要求
        min_count = math.ceil(proportion * num_patches)
        min_patch_required[class_index] = min_count

    # 4) 计算"剩余patch数" = 超过阈值的patch数 - 最小要求的patch数
    remaining_class_count = {
        class_index: max(0, count - min_patch_required.get(class_index, 0))
        for class_index, count in class_count.items()
    }

    # 5) 筛选剩余值 > 0 的类别
    remaining_classes = {
        class_index: remaining_class_count[class_index]
        for class_index in remaining_class_count
        if remaining_class_count[class_index] > 0
    }

    # 6) 确定最终类别 final_class
    if len(remaining_classes) > 0:
        # 找到最大的 remaining_class_count
        max_remaining_count = max(remaining_classes.values())
        # 找到所有 remaining_class_count 等于最大值的类别（可能不止一个）
        classes_with_max_count = [
            class_index for class_index, c_val in remaining_classes.items()
            if c_val == max_remaining_count
        ]
        # 如果有多个并列最大，则比较这些类别的概率和
        if len(classes_with_max_count) > 1:
            final_class = max(classes_with_max_count, key=lambda idx: class_prob_sum[idx])
        else:
            final_class = classes_with_max_count[0]
    else:
        # 如果所有类别都没能满足 min_window_counts 的要求
        # 那么直接选 "概率和" 最大的类别
        final_class = max(class_prob_sum, key=class_prob_sum.get)

    # 7) 计算 pred_probs 用于 AUC
    # 对每个类别，用 "超过阈值的概率之和" 作为原始得分
    # 然后做归一化
    scores = np.array([class_prob_sum[i] for i in sorted(class_prob_sum.keys())])
    total = scores.sum()
    if total > 0:
        pred_probs = scores / total
    else:
        # 若所有概率和都为 0，则给均匀分布
        pred_probs = np.ones(num_classes) / num_classes

    return final_class, pred_probs


# =============================================================================
# CF-BGG策略实现：Counterfactual-Banzhaf-Gear Graph Aggregation（二分类版本）
# =============================================================================

def _apply_cf_bgg_strategy(patch_probs, K=10, M=2, alpha=0.6, S=16, sigma=1.0,
                          lambda_=1.0, beta=0.6, tau_m=0.20, tau_kappa=0.4,
                          tau_f=0.10, gamma=1.0, eps=1e-6, random_seed=42):
    """
    CF-BGG策略：融合反事实影响、Banzhaf交互、齿轮图一致性的个体级聚合（二分类版本）

    Args:
        patch_probs: patch级预测概率 (n, 2)
        K: 每类Top-K证据数
        M: 多样性覆盖参数
        alpha: LOO vs Banzhaf权重 (0~1)
        S: 联盟采样数
        sigma: 相似度带宽
        lambda_: 一致性放大系数
        beta: 对比抑制系数
        tau_m: margin阈值
        tau_kappa: 一致性阈值
        tau_f: 支持率阈值
        gamma: 温度参数
        eps: 数值稳定常数

    Returns:
        final_class: 最终预测类别
        pred_probs: 预测概率分布
    """
    import numpy as np
    import math

    # 设置随机种子确保结果可重现
    np.random.seed(random_seed)

    # 安全性检查
    patch_probs = np.clip(patch_probs, eps, 1-eps)
    n, C = patch_probs.shape

    if C != 2:
        raise ValueError("CF-BGG二分类版本要求C=2")

    # Step 1: 计算基线对数几率（二分类：只需正类logit）
    ell = _compute_log_odds_binary(patch_probs, eps)

    # Step 2: 计算反事实影响与可靠度权重（二分类：只对正类）
    phi1, r_weights = _compute_counterfactual_influence_binary(patch_probs, ell, eps)

    # Step 3: 计算Banzhaf交互增益（二分类：复杂度降C倍）
    phiB = _compute_banzhaf_interaction_binary(patch_probs, ell, S, eps)

    # Step 4: 组合LOO和Banzhaf证据
    phi = alpha * phi1 + (1 - alpha) * phiB

    # Step 5: 一致性图重加权（二分类：使用1维相似度）
    phi_tilde = _apply_gear_reweighting_binary(phi, patch_probs, sigma, lambda_)

    # Step 6: 证据汇总与对比（二分类：直接用margin）
    F0, F1, kappa, support_rates = _aggregate_evidence_scores_binary(
        phi_tilde, ell, patch_probs, K, M, beta, n
    )

    # Step 7: 置信判据与弃权检查（二分类：用标量margin）
    margin = F1 - F0

    final_class = 1 if margin > 0 else 0

    # 弃权条件检查
    if (abs(margin) < tau_m or kappa < tau_kappa or
        support_rates[final_class] < tau_f):
        # 弃权情况：返回均匀分布
        pred_probs = np.array([0.5, 0.5])
    else:
        # 正常情况：sigmoid映射
        q = 1.0 / (1.0 + np.exp(-gamma * margin))
        pred_probs = np.array([1-q, q])

    return int(final_class), pred_probs


def _compute_log_odds_binary(patch_probs, eps):
    """计算基线对数几率（二分类：只需正类logit）"""
    import numpy as np

    # 平均概率
    p_bar = np.mean(patch_probs, axis=0)
    p_bar = np.clip(p_bar, eps, 1-eps)

    # 二分类：只需正类对数几率
    ell = np.log(p_bar[1] / (1 - p_bar[1] + eps))

    return ell


def _compute_counterfactual_influence_binary(patch_probs, ell, eps):
    """计算反事实影响与可靠度权重（二分类版本）"""
    import numpy as np

    n = len(patch_probs)
    phi1 = np.zeros(n)  # 只需一列（正类）
    r_weights = np.zeros(n)

    for i in range(n):
        # 计算leave-one-out对数几率
        if n <= 1:
            # 单个patch的情况，无法计算leave-one-out
            ell_minus_i = ell  # 使用原始值
        else:
            p_bar_minus_i = (np.sum(patch_probs, axis=0) - patch_probs[i]) / (n - 1)
            p_bar_minus_i = np.clip(p_bar_minus_i, eps, 1-eps)
            ell_minus_i = np.log(p_bar_minus_i[1] / (1 - p_bar_minus_i[1] + eps))

        # 反事实影响 (只对正类)
        phi1[i] = ell - ell_minus_i

        # 可靠度权重（二分类：使用|2p1-1|）
        H_i = -np.sum(patch_probs[i] * np.log(patch_probs[i] + eps))  # 熵
        margin_i = abs(2 * patch_probs[i, 1] - 1)  # 二分类margin
        r_weights[i] = (1 - H_i / np.log(2)) * margin_i

    # 可靠度权重应用
    phi1 = phi1 * r_weights

    return phi1, r_weights


def _compute_banzhaf_interaction_binary(patch_probs, ell, S, eps):
    """计算Banzhaf交互增益（二分类版本：复杂度降一半）"""
    import numpy as np

    n = len(patch_probs)
    phiB = np.zeros(n)  # 只需一列

    # 性能优化：限制采样数和patch数
    if n > 50:
        # 大数据集：使用采样策略
        max_samples = min(20, n // 3)
        selected_indices = np.random.choice(n, max_samples, replace=False)
        patch_probs_sampled = patch_probs[selected_indices]
        phiB_sampled = np.zeros(max_samples)

        # 减少采样轮数
        S_reduced = min(S, 8)

        for _ in range(S_reduced):
            for idx, i in enumerate(selected_indices):
                others = [j for j in range(max_samples) if j != idx]
                if len(others) == 0:
                    continue

                coalition_size = min(np.random.randint(1, max(2, len(others) // 2)), 3)
                U = np.random.choice(others, coalition_size, replace=False)

                # 二分类：只计算正类logit
                p_with_i = np.mean(np.append(patch_probs_sampled[U, 1], patch_probs_sampled[idx, 1]))
                p_without_i = np.mean(patch_probs_sampled[U, 1]) if len(U) > 0 else np.mean(patch_probs_sampled[:, 1])

                p_with_i = np.clip(p_with_i, eps, 1-eps)
                p_without_i = np.clip(p_without_i, eps, 1-eps)

                L_with_i = np.log(p_with_i / (1 - p_with_i + eps))
                L_without_i = np.log(p_without_i / (1 - p_without_i + eps))

                delta = L_with_i - L_without_i
                phiB_sampled[idx] += max(0, delta)

        phiB_sampled = phiB_sampled / S_reduced

        # 扩展结果到全集
        phiB[selected_indices] = phiB_sampled

        # 为未采样的patch使用近似值
        for i in range(n):
            if i not in selected_indices:
                # 使用最近邻的值
                distances = np.abs(patch_probs[i, 1] - patch_probs[selected_indices, 1])
                nearest_idx = selected_indices[np.argmin(distances)]
                nearest_sampled_idx = np.where(selected_indices == nearest_idx)[0][0]
                phiB[i] = phiB_sampled[nearest_sampled_idx] * 0.8  # 稍微降权

    else:
        # 小数据集：使用简化计算
        S_reduced = min(S, 12)

        for _ in range(S_reduced):
            for i in range(n):
                others = [j for j in range(n) if j != i]
                if len(others) == 0:
                    continue

                coalition_size = min(np.random.randint(1, max(2, len(others) // 2)), 4)
                U = np.random.choice(others, coalition_size, replace=False)

                # 二分类：只计算正类logit
                p_with_i = np.mean(np.append(patch_probs[U, 1], patch_probs[i, 1]))
                p_without_i = np.mean(patch_probs[U, 1])

                p_with_i = np.clip(p_with_i, eps, 1-eps)
                p_without_i = np.clip(p_without_i, eps, 1-eps)

                L_with_i = np.log(p_with_i / (1 - p_with_i + eps))
                L_without_i = np.log(p_without_i / (1 - p_without_i + eps))

                delta = L_with_i - L_without_i
                phiB[i] += max(0, delta)

        phiB = phiB / S_reduced

    return phiB


def _apply_gear_reweighting_binary(phi, patch_probs, sigma, lambda_):
    """一致性图重加权（二分类版本：使用1维相似度）"""
    import numpy as np

    n = len(phi)
    phi_tilde = phi.copy()

    # 性能优化：大数据集使用k-NN近似
    if n > 30:
        # 使用k近邻而非全连接图
        k = min(10, n // 3)

        for i in range(n):
            # 计算与其他patch的距离（二分类：使用1维差）
            distances = np.abs(patch_probs[:, 1] - patch_probs[i, 1])

            # 找k个最近邻（排除自己）
            neighbor_indices = np.argsort(distances)[1:k+1]
            neighbor_distances = distances[neighbor_indices]

            # 计算相似度（只对k个邻居）
            similarities = np.exp(-neighbor_distances**2 / (sigma**2))

            # 图重加权（只考虑k个邻居）
            positive_neighbors = np.sum(similarities * (phi[neighbor_indices] > 0))
            total_weight = np.sum(similarities)

            if total_weight > 0:
                consistency_boost = lambda_ * positive_neighbors / (1 + total_weight)
                phi_tilde[i] = phi[i] * (1 + consistency_boost)

    else:
        # 小数据集：使用完整计算但优化实现
        # 向量化距离计算
        diff_matrix = np.abs(patch_probs[:, 1:2] - patch_probs[:, 1:2].T)

        # 计算相似度矩阵
        similarities = np.exp(-diff_matrix**2 / (sigma**2))
        np.fill_diagonal(similarities, 0)  # 去除自环

        # 向量化图重加权
        positive_mask = phi > 0
        positive_neighbors = similarities @ positive_mask
        total_weights = np.sum(similarities, axis=1)

        # 避免除零
        valid_mask = total_weights > 0
        consistency_boost = np.zeros(n)
        consistency_boost[valid_mask] = lambda_ * positive_neighbors[valid_mask] / (1 + total_weights[valid_mask])

        phi_tilde = phi * (1 + consistency_boost)

    return phi_tilde


def _aggregate_evidence_scores_binary(phi_tilde, ell, patch_probs, K, M, beta, n):
    """证据汇总与对比式评分（二分类版本：使用标量margin）"""
    import numpy as np

    # 为正类和负类选择Top-K证据
    positive_scores = phi_tilde[phi_tilde > 0]
    negative_scores = phi_tilde[phi_tilde < 0]

    # 选择Top-K正证据
    if len(positive_scores) > 0:
        top_k_count = min(K, len(positive_scores))
        positive_indices = np.where(phi_tilde > 0)[0]
        top_positive_indices = positive_indices[np.argpartition(phi_tilde[positive_indices], -top_k_count)[-top_k_count:]]
        S_positive = top_positive_indices.tolist()
    else:
        S_positive = []

    # 选择Top-K负证据（对负类有利的证据）
    if len(negative_scores) > 0:
        top_k_count = min(K, len(negative_scores))
        negative_indices = np.where(phi_tilde < 0)[0]
        top_negative_indices = negative_indices[np.argpartition(-phi_tilde[negative_indices], -top_k_count)[-top_k_count:]]
        S_negative = top_negative_indices.tolist()
    else:
        S_negative = []

    support_rates = np.array([len(S_negative) / n, len(S_positive) / n])

    # 计算F0和F1
    # F1（正类）：基线 + 正证据 - β×负证据
    F1 = ell
    if S_positive:
        F1 += np.sum(phi_tilde[S_positive])
    if S_negative:
        F1 -= beta * np.sum(abs(phi_tilde[S_negative]))  # 负证据的抑制

    # F0（负类）：对称计算
    F0 = -ell
    if S_negative:
        F0 += np.sum(abs(phi_tilde[S_negative]))
    if S_positive:
        F0 -= beta * np.sum(phi_tilde[S_positive])

    # 简化的一致性指标计算
    consistency_values = []

    # 计算正证据的一致性
    if len(S_positive) > 1:
        sample_size = min(3, len(S_positive))
        sampled_indices = S_positive[:sample_size]
        if len(sampled_indices) >= 2:
            similarities = []
            for i in range(len(sampled_indices)):
                for j in range(i + 1, len(sampled_indices)):
                    idx_i, idx_j = sampled_indices[i], sampled_indices[j]
                    diff = abs(patch_probs[idx_i, 1] - patch_probs[idx_j, 1])
                    similarity = np.exp(-diff)
                    similarities.append(similarity)
            if similarities:
                consistency_values.append(np.mean(similarities))

    # 计算负证据的一致性
    if len(S_negative) > 1:
        sample_size = min(3, len(S_negative))
        sampled_indices = S_negative[:sample_size]
        if len(sampled_indices) >= 2:
            similarities = []
            for i in range(len(sampled_indices)):
                for j in range(i + 1, len(sampled_indices)):
                    idx_i, idx_j = sampled_indices[i], sampled_indices[j]
                    diff = abs(patch_probs[idx_i, 1] - patch_probs[idx_j, 1])
                    similarity = np.exp(-diff)
                    similarities.append(similarity)
            if similarities:
                consistency_values.append(np.mean(similarities))

    kappa = np.mean(consistency_values) if consistency_values else 0.0

    return F0, F1, kappa, support_rates


def _calc_subject_metrics(pred_labels, pred_probs, true_labels, subject_ids, classes_names,
                         method='majority_vote', confidence_threshold=0.6):
    """
    计算个体级分类指标（二分类版本）

    Args:
        pred_labels: patch级预测标签
        pred_probs: patch级预测概率
        true_labels: patch级真实标签
        subject_ids: patch对应的个体ID
        classes_names: 类名列表
        method: 聚合方法
        confidence_threshold: 置信度阈值

    Returns:
        tuple: 与_calc_metrics相同格式的个体级指标
    """
    # 聚合为个体级预测
    subj_pred_labels, subj_true_labels, subj_pred_probs, _ = aggregate_subject_predictions(
        pred_labels, pred_probs, true_labels, subject_ids, method, confidence_threshold
    )

    # 计算个体级指标
    return _calc_metrics(subj_pred_labels, subj_true_labels, subj_pred_probs, classes_names)


# ==========
# 逐类与宏/微统计表（所有可度量列均为百分数；support 为样本数）
# ==========
def _save_metrics(pred_labels, true_labels, pred_probs, log_dir, class_names, file_name, model_name,
                  subj_pred_labels=None, subj_true_labels=None, subj_pred_probs=None,
                  subject_results_all=None):
    import pandas as pd

    def save_confusion_matrix_image(cm, labels, path, title):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    # Fixed black annotations remain legible on the light
                    # cells used by the IS-versus-HS experiment and satisfy
                    # the publication figure style for this binary task.
                    color="black",
                    fontweight="semibold",
                )

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)
    n_classes = len(class_names)

    # 若为二分类且 pred_probs 为一维数组，则视作正类概率并堆成两列
    if n_classes == 2 and pred_probs is not None and np.ndim(pred_probs) == 1:
        pred_probs = np.vstack([1 - pred_probs, pred_probs]).T

    # 生成 classification_report 并转 DataFrame
    # 注意：sklearn 原本 precision/recall/f1 为 0~1；这里统一乘 100
    report = classification_report(
        true_labels, pred_labels, target_names=class_names, digits=6, output_dict=True, zero_division=0
    )
    class_metrics = pd.DataFrame(report).transpose()

    # 将 precision/recall/f1-score 统一转为百分数
    for col in ["precision", "recall", "f1-score"]:
        if col in class_metrics.columns:
            class_metrics[col] = class_metrics[col].astype(float) * 100.0
    # >>> 新增：提前创建空的 MCC 列，防止后续列筛选时被忽略
    if "MCC" not in class_metrics.columns:
        class_metrics["MCC"] = np.nan

    # 每类 AUC / AUCPR（百分数）
    auc_values, aucpr_values = [], []
    if pred_probs is not None:
        for i, _ in enumerate(class_names):
            class_true = (true_labels == i).astype(int)
            try:
                auc = roc_auc_score(class_true, pred_probs[:, i]) * 100.0
            except Exception:
                auc = 0.0
            auc_values.append(auc)

            try:
                ap = average_precision_score(class_true, pred_probs[:, i]) * 100.0
            except Exception:
                ap = 0.0
            aucpr_values.append(ap)
        class_metrics.loc[class_names, "AUC"] = auc_values
        class_metrics.loc[class_names, "AUCPR"] = aucpr_values
    else:
        class_metrics.loc[class_names, "AUC"] = 0.0
        class_metrics.loc[class_names, "AUCPR"] = 0.0

    # 混淆矩阵（固定按 0..K-1 顺序）
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))
    total = cm.sum()

    # 逐类（one-vs-rest）G-Mean / Accuracy（百分数）
    g_means, accuracies = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - (tp + fn + fp)

        sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0        # 该类 TPR
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # 该类 TNR
        g_means.append(np.sqrt(sens_i * spec_i) * 100.0)
        acc_i = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        accuracies.append(acc_i * 100.0)

    class_metrics.loc[class_names, "G-Mean"] = g_means
    class_metrics.loc[class_names, "Accuracy"] = accuracies

    # 以标签1为正类的整体 G-Mean（用于 micro 的参考——二分类情形等价）
    if n_classes == 2:
        TN, FP, FN, TP = _binary_confusion_counts(true_labels, pred_labels)
        sens_pos = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec_pos = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        gmean_pos_pct = np.sqrt(sens_pos * spec_pos) * 100.0
    else:
        gmean_pos_pct = np.nan

    # 宏平均（macro avg）
    macro_auc = float(np.mean(auc_values)) if len(auc_values) > 0 else 0.0
    macro_aucpr = float(np.mean(aucpr_values)) if len(aucpr_values) > 0 else 0.0
    macro_g_mean = float(np.nanmean(class_metrics.loc[class_names, "G-Mean"])) if len(class_names) > 0 else 0.0
    macro_accuracy = accuracy_score(true_labels, pred_labels) * 100.0  # 这里保持与原实现一致（总体准确率）

    class_metrics.loc["macro avg", "AUC"] = macro_auc
    class_metrics.loc["macro avg", "AUCPR"] = macro_aucpr
    class_metrics.loc["macro avg", "G-Mean"] = macro_g_mean
    class_metrics.loc["macro avg", "Accuracy"] = macro_accuracy
    # macro 的 precision/recall/f1-score 已在前面整体 *100 了

    # 微平均（micro avg）——precision/recall/f1/G-Mean/Accuracy/AUC/AUCPR/MCC（百分数）
    micro_precision_val = precision_score(true_labels, pred_labels, average="micro", zero_division=0) * 100.0
    micro_recall_val = recall_score(true_labels, pred_labels, average="micro", zero_division=0) * 100.0
    micro_f1_val = f1_score(true_labels, pred_labels, average="micro", zero_division=0) * 100.0
    micro_accuracy_val = accuracy_score(true_labels, pred_labels) * 100.0

    # micro AUC / AUCPR：多类用 one-vs-rest 的 'micro' 定义
    if pred_probs is not None:
        if n_classes == 2:
            # 与原实现保持一致：二分类直接用正类分数
            try:
                micro_auc = roc_auc_score(true_labels, pred_probs[:, 1]) * 100.0
            except Exception:
                micro_auc = 0.0
            try:
                micro_aucpr = average_precision_score(true_labels, pred_probs[:, 1]) * 100.0
            except Exception:
                micro_aucpr = 0.0
        else:
            # 多类：需要二值化标签
            y_true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
            try:
                micro_auc = roc_auc_score(y_true_bin, pred_probs, average="micro", multi_class="ovr") * 100.0
            except Exception:
                micro_auc = 0.0
            try:
                micro_aucpr = average_precision_score(y_true_bin, pred_probs, average="micro") * 100.0
            except Exception:
                micro_aucpr = 0.0
    else:
        micro_auc = 0.0
        micro_aucpr = 0.0

    if n_classes == 2:
        micro_g_mean = gmean_pos_pct
    else:
        # 多类 micro 的 G-Mean 没有统一定义，这里置为 NaN 以避免误导
        micro_g_mean = np.nan

    try:
        micro_mcc = matthews_corrcoef(true_labels, pred_labels) * 100.0
    except Exception:
        micro_mcc = 0.0

    class_metrics.loc["micro avg"] = {
        "precision": micro_precision_val,
        "recall": micro_recall_val,
        "f1-score": micro_f1_val,
        "AUC": micro_auc,
        "AUCPR": micro_aucpr,
        "G-Mean": micro_g_mean,
        "Accuracy": micro_accuracy_val,
        "MCC": micro_mcc,
        "support": cm.sum(),   # support 保持样本数（非百分数）
    }

    # 删除不需要的行
    class_metrics = class_metrics.drop(index="accuracy", errors="ignore")
    class_metrics = class_metrics.drop(index="weighted avg", errors="ignore")

    # 重新排列列顺序（缺失的列会自动忽略）
    cols = [
        "precision", "recall", "f1-score",
        "AUC", "AUCPR",
        "G-Mean", "Accuracy",
        "MCC",
        "support", "value"
    ]
    class_metrics = class_metrics[[c for c in cols if c in class_metrics.columns]]

    # 保存到 Excel
    os.makedirs(log_dir, exist_ok=True)
    save_confusion_matrix_image(
        cm,
        class_names,
        os.path.join(log_dir, f"{file_name}_{model_name}_confusion_matrix.png"),
        f"{file_name} {model_name} confusion matrix",
    )
    file_path = os.path.join(log_dir, f"{file_name}_{model_name}.xlsx")
    with pd.ExcelWriter(file_path) as writer:
        class_metrics.to_excel(writer, sheet_name="Patch_Metrics")

        # 如果提供了个体级别的指标，保存到第二个Sheet
        if (subj_pred_labels is not None and subj_true_labels is not None
            and subj_pred_probs is not None):

            # 转换为 NumPy
            subj_pred_labels = np.array(subj_pred_labels).astype(int)
            subj_true_labels = np.array(subj_true_labels).astype(int)
            subj_n_classes = len(class_names)

            # 若为二分类且 subj_pred_probs 为一维数组，则视作正类概率并堆成两列
            if subj_n_classes == 2 and subj_pred_probs is not None and np.ndim(subj_pred_probs) == 1:
                subj_pred_probs = np.vstack([1 - subj_pred_probs, subj_pred_probs]).T

            # 生成个体级别的 classification_report
            subj_report = classification_report(
                subj_true_labels, subj_pred_labels, target_names=class_names, digits=6, output_dict=True, zero_division=0
            )
            subj_metrics = pd.DataFrame(subj_report).transpose()

            # 将 precision/recall/f1-score 转为百分数
            for col in ["precision", "recall", "f1-score"]:
                if col in subj_metrics.columns:
                    subj_metrics[col] = subj_metrics[col].astype(float) * 100.0

            # 提前创建空的 MCC 列
            if "MCC" not in subj_metrics.columns:
                subj_metrics["MCC"] = np.nan

            # 每类 AUC / AUCPR（百分数）
            subj_auc_values, subj_aucpr_values = [], []
            if subj_pred_probs is not None:
                for i, _ in enumerate(class_names):
                    class_true = (subj_true_labels == i).astype(int)
                    try:
                        auc = roc_auc_score(class_true, subj_pred_probs[:, i]) * 100.0
                    except Exception:
                        auc = 0.0
                    subj_auc_values.append(auc)

                    try:
                        ap = average_precision_score(class_true, subj_pred_probs[:, i]) * 100.0
                    except Exception:
                        ap = 0.0
                    subj_aucpr_values.append(ap)
                subj_metrics.loc[class_names, "AUC"] = subj_auc_values
                subj_metrics.loc[class_names, "AUCPR"] = subj_aucpr_values
            else:
                subj_metrics.loc[class_names, "AUC"] = 0.0
                subj_metrics.loc[class_names, "AUCPR"] = 0.0

            # 混淆矩阵
            subj_cm = confusion_matrix(subj_true_labels, subj_pred_labels, labels=list(range(subj_n_classes)))
            subj_total = subj_cm.sum()

            # 逐类 G-Mean / Accuracy
            subj_g_means, subj_accuracies = [], []
            for i in range(subj_n_classes):
                tp = subj_cm[i, i]
                fn = np.sum(subj_cm[i, :]) - tp
                fp = np.sum(subj_cm[:, i]) - tp
                tn = subj_total - (tp + fn + fp)

                sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                subj_g_means.append(np.sqrt(sens_i * spec_i) * 100.0)
                acc_i = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                subj_accuracies.append(acc_i * 100.0)

            subj_metrics.loc[class_names, "G-Mean"] = subj_g_means
            subj_metrics.loc[class_names, "Accuracy"] = subj_accuracies

            # 以标签1为正类的整体 G-Mean（二分类情形）
            if subj_n_classes == 2:
                subj_TN, subj_FP, subj_FN, subj_TP = _binary_confusion_counts(subj_true_labels, subj_pred_labels)
                subj_sens_pos = subj_TP / (subj_TP + subj_FN) if (subj_TP + subj_FN) > 0 else 0.0
                subj_spec_pos = subj_TN / (subj_TN + subj_FP) if (subj_TN + subj_FP) > 0 else 0.0
                subj_gmean_pos_pct = np.sqrt(subj_sens_pos * subj_spec_pos) * 100.0
            else:
                subj_gmean_pos_pct = np.nan

            # 宏平均
            subj_macro_auc = float(np.mean(subj_auc_values)) if len(subj_auc_values) > 0 else 0.0
            subj_macro_aucpr = float(np.mean(subj_aucpr_values)) if len(subj_aucpr_values) > 0 else 0.0
            subj_macro_g_mean = float(np.nanmean(subj_metrics.loc[class_names, "G-Mean"])) if len(class_names) > 0 else 0.0
            subj_macro_accuracy = accuracy_score(subj_true_labels, subj_pred_labels) * 100.0

            subj_metrics.loc["macro avg", "AUC"] = subj_macro_auc
            subj_metrics.loc["macro avg", "AUCPR"] = subj_macro_aucpr
            subj_metrics.loc["macro avg", "G-Mean"] = subj_macro_g_mean
            subj_metrics.loc["macro avg", "Accuracy"] = subj_macro_accuracy

            # 微平均
            subj_micro_precision_val = precision_score(subj_true_labels, subj_pred_labels, average="micro", zero_division=0) * 100.0
            subj_micro_recall_val = recall_score(subj_true_labels, subj_pred_labels, average="micro", zero_division=0) * 100.0
            subj_micro_f1_val = f1_score(subj_true_labels, subj_pred_labels, average="micro", zero_division=0) * 100.0
            subj_micro_accuracy_val = accuracy_score(subj_true_labels, subj_pred_labels) * 100.0

            # micro AUC / AUCPR
            if subj_pred_probs is not None:
                if subj_n_classes == 2:
                    try:
                        subj_micro_auc = roc_auc_score(subj_true_labels, subj_pred_probs[:, 1]) * 100.0
                    except Exception:
                        subj_micro_auc = 0.0
                    try:
                        subj_micro_aucpr = average_precision_score(subj_true_labels, subj_pred_probs[:, 1]) * 100.0
                    except Exception:
                        subj_micro_aucpr = 0.0
                else:
                    subj_y_true_bin = label_binarize(subj_true_labels, classes=list(range(subj_n_classes)))
                    try:
                        subj_micro_auc = roc_auc_score(subj_y_true_bin, subj_pred_probs, average="micro", multi_class="ovr") * 100.0
                    except Exception:
                        subj_micro_auc = 0.0
                    try:
                        subj_micro_aucpr = average_precision_score(subj_y_true_bin, subj_pred_probs, average="micro") * 100.0
                    except Exception:
                        subj_micro_aucpr = 0.0
            else:
                subj_micro_auc = 0.0
                subj_micro_aucpr = 0.0

            if subj_n_classes == 2:
                subj_micro_g_mean = subj_gmean_pos_pct
            else:
                subj_micro_g_mean = np.nan

            try:
                subj_micro_mcc = matthews_corrcoef(subj_true_labels, subj_pred_labels) * 100.0
            except Exception:
                subj_micro_mcc = 0.0

            subj_metrics.loc["micro avg"] = {
                "precision": subj_micro_precision_val,
                "recall": subj_micro_recall_val,
                "f1-score": subj_micro_f1_val,
                "AUC": subj_micro_auc,
                "AUCPR": subj_micro_aucpr,
                "G-Mean": subj_micro_g_mean,
                "Accuracy": subj_micro_accuracy_val,
                "MCC": subj_micro_mcc,
                "support": subj_cm.sum(),
            }

            # 删除不需要的行
            subj_metrics = subj_metrics.drop(index="accuracy", errors="ignore")
            subj_metrics = subj_metrics.drop(index="weighted avg", errors="ignore")

            # 重新排列列顺序
            subj_metrics = subj_metrics[[c for c in cols if c in subj_metrics.columns]]

            # 保存到第二个Sheet
            subj_metrics.to_excel(writer, sheet_name="Subject_Metrics")

        # 如果提供了多策略个体级结果，为每个策略创建独立的Sheet
        if subject_results_all:
            for method, results in subject_results_all.items():
                if (results and 'pred_labels' in results and 'true_labels' in results
                    and 'pred_probs' in results and results['pred_labels'] is not None
                    and results['true_labels'] is not None and results['pred_probs'] is not None):

                    # 转换为 NumPy
                    method_pred_labels = np.array(results['pred_labels']).astype(int)
                    method_true_labels = np.array(results['true_labels']).astype(int)
                    method_pred_probs = results['pred_probs']
                    method_n_classes = len(class_names)

                    # 若为二分类且 method_pred_probs 为一维数组，则视作正类概率并堆成两列
                    if method_n_classes == 2 and method_pred_probs is not None and np.ndim(method_pred_probs) == 1:
                        method_pred_probs = np.vstack([1 - method_pred_probs, method_pred_probs]).T

                    # 生成该策略的 classification_report
                    method_report = classification_report(
                        method_true_labels, method_pred_labels, target_names=class_names, digits=6, output_dict=True, zero_division=0
                    )
                    method_metrics = pd.DataFrame(method_report).transpose()

                    # 将 precision/recall/f1-score 转为百分数
                    for col in ["precision", "recall", "f1-score"]:
                        if col in method_metrics.columns:
                            method_metrics[col] = method_metrics[col].astype(float) * 100.0

                    # 提前创建空的 MCC 列
                    if "MCC" not in method_metrics.columns:
                        method_metrics["MCC"] = np.nan

                    # 每类 AUC / AUCPR（百分数）
                    method_auc_values, method_aucpr_values = [], []
                    if method_pred_probs is not None:
                        for i, _ in enumerate(class_names):
                            class_true = (method_true_labels == i).astype(int)
                            try:
                                auc = roc_auc_score(class_true, method_pred_probs[:, i]) * 100.0
                            except Exception:
                                auc = 0.0
                            method_auc_values.append(auc)

                            try:
                                ap = average_precision_score(class_true, method_pred_probs[:, i]) * 100.0
                            except Exception:
                                ap = 0.0
                            method_aucpr_values.append(ap)
                        method_metrics.loc[class_names, "AUC"] = method_auc_values
                        method_metrics.loc[class_names, "AUCPR"] = method_aucpr_values
                    else:
                        method_metrics.loc[class_names, "AUC"] = 0.0
                        method_metrics.loc[class_names, "AUCPR"] = 0.0

                    # 混淆矩阵
                    method_cm = confusion_matrix(method_true_labels, method_pred_labels, labels=list(range(method_n_classes)))
                    method_total = method_cm.sum()

                    # 逐类（one-vs-rest）G-Mean / Accuracy（百分数）
                    method_g_means, method_accuracies = [], []
                    for i in range(method_n_classes):
                        tp = method_cm[i, i]
                        fn = np.sum(method_cm[i, :]) - tp
                        fp = np.sum(method_cm[:, i]) - tp
                        tn = method_total - (tp + fn + fp)

                        sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0        # 该类 TPR
                        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # 该类 TNR
                        method_g_means.append(np.sqrt(sens_i * spec_i) * 100.0)
                        acc_i = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                        method_accuracies.append(acc_i * 100.0)

                    method_metrics.loc[class_names, "G-Mean"] = method_g_means
                    method_metrics.loc[class_names, "Accuracy"] = method_accuracies

                    # 宏平均
                    method_macro_auc = float(np.mean(method_auc_values)) if len(method_auc_values) > 0 else 0.0
                    method_macro_aucpr = float(np.mean(method_aucpr_values)) if len(method_aucpr_values) > 0 else 0.0
                    method_macro_g_mean = float(np.nanmean(method_metrics.loc[class_names, "G-Mean"])) if len(class_names) > 0 else 0.0
                    method_macro_accuracy = accuracy_score(method_true_labels, method_pred_labels) * 100.0

                    method_metrics.loc["macro avg", "AUC"] = method_macro_auc
                    method_metrics.loc["macro avg", "AUCPR"] = method_macro_aucpr
                    method_metrics.loc["macro avg", "G-Mean"] = method_macro_g_mean
                    method_metrics.loc["macro avg", "Accuracy"] = method_macro_accuracy

                    # 微平均
                    method_micro_precision_val = precision_score(method_true_labels, method_pred_labels, average="micro", zero_division=0) * 100.0
                    method_micro_recall_val = recall_score(method_true_labels, method_pred_labels, average="micro", zero_division=0) * 100.0
                    method_micro_f1_val = f1_score(method_true_labels, method_pred_labels, average="micro", zero_division=0) * 100.0
                    method_micro_accuracy_val = accuracy_score(method_true_labels, method_pred_labels) * 100.0

                    # micro AUC / AUCPR：多类用 one-vs-rest 的 'micro' 定义
                    if method_pred_probs is not None:
                        if method_n_classes == 2:
                            try:
                                method_micro_auc = roc_auc_score(method_true_labels, method_pred_probs[:, 1]) * 100.0
                            except Exception:
                                method_micro_auc = 0.0
                            try:
                                method_micro_aucpr = average_precision_score(method_true_labels, method_pred_probs[:, 1]) * 100.0
                            except Exception:
                                method_micro_aucpr = 0.0
                        else:
                            method_y_true_bin = label_binarize(method_true_labels, classes=list(range(method_n_classes)))
                            try:
                                method_micro_auc = roc_auc_score(method_y_true_bin, method_pred_probs, average="micro", multi_class="ovr") * 100.0
                            except Exception:
                                method_micro_auc = 0.0
                            try:
                                method_micro_aucpr = average_precision_score(method_y_true_bin, method_pred_probs, average="micro") * 100.0
                            except Exception:
                                method_micro_aucpr = 0.0
                    else:
                        method_micro_auc = 0.0
                        method_micro_aucpr = 0.0

                    # 计算 micro G-Mean
                    if method_n_classes == 2:
                        method_TN, method_FP, method_FN, method_TP = _binary_confusion_counts(method_true_labels, method_pred_labels)
                        method_sens_pos = method_TP / (method_TP + method_FN) if (method_TP + method_FN) > 0 else 0.0
                        method_spec_pos = method_TN / (method_TN + method_FP) if (method_TN + method_FP) > 0 else 0.0
                        method_micro_g_mean = np.sqrt(method_sens_pos * method_spec_pos) * 100.0
                    else:
                        method_micro_g_mean = np.nan

                    try:
                        method_micro_mcc = matthews_corrcoef(method_true_labels, method_pred_labels) * 100.0
                    except Exception:
                        method_micro_mcc = 0.0

                    method_metrics.loc["micro avg"] = {
                        "precision": method_micro_precision_val,
                        "recall": method_micro_recall_val,
                        "f1-score": method_micro_f1_val,
                        "AUC": method_micro_auc,
                        "AUCPR": method_micro_aucpr,
                        "G-Mean": method_micro_g_mean,
                        "Accuracy": method_micro_accuracy_val,
                        "MCC": method_micro_mcc,
                        "support": method_cm.sum(),   # support 保持样本数（非百分数）
                    }

                    # 删除不需要的行
                    method_metrics = method_metrics.drop(index="accuracy", errors="ignore")
                    method_metrics = method_metrics.drop(index="weighted avg", errors="ignore")

                    # 重新排列列顺序（缺失的列会自动忽略）
                    method_metrics = method_metrics[[c for c in cols if c in method_metrics.columns]]

                    # 保存到独立的Sheet
                    sheet_name = f"Subject_{method}"
                    method_metrics.to_excel(writer, sheet_name=sheet_name)


import collections


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, abc.Mapping):  # 更新这里
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, abc.Sequence):  # 和这里
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list; found {type(input)}")



def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"configs/data_configs.py", os.path.join(destination_dir, f"data_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"trainer.py", os.path.join(destination_dir, f"trainer.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))


def _plot_umap(model, data_loader, device, save_dir):
    import umap
    import umap.plot
    from matplotlib.colors import ListedColormap
    classes_names = ['N','S','V','F','Q']

    font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 17}
    plt.rc('font', **font)

    with torch.no_grad():
        # Source flow
        data = data_loader.dataset.x_data.float().to(device)
        labels = data_loader.dataset.y_data.view((-1)).long()
        out = model[0](data)
        features = model[1](out)


    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))

    #cmaps = plt.get_cmap('jet')
    model_reducer = umap.UMAP() #n_neighbors=3, min_dist=0.3, metric='correlation', random_state=42)
    embedding = model_reducer.fit_transform(features.detach().cpu().numpy())

    # Normalize the labels to [0, 1] for colormap
    norm_labels = labels / 4.0


    # Create a new colormap by extracting the first 5 colors from "Paired"
    paired = plt.cm.get_cmap('Paired', 12)  # 12 distinct colors
    new_colors = [paired(0), paired(1), paired(2), paired(4), paired(6)]  # Skip every second color, but take both from the first pair
    new_cmap = ListedColormap(new_colors)

    print("Plotting UMAP ...")
    plt.figure(figsize=(16, 10))
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels,  s=10, cmap='Spectral')
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norm_labels, cmap=new_cmap, s=15)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names,  title="Classes")
    file_name = "umap_.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()
