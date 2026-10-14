import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import os
import collections
import numpy as np

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class

from utils_twoclassify1 import AverageMeter, to_device, _save_metrics, save_checkpoint
from utils_twoclassify1 import fix_randomness, starting_logs, _calc_metrics

from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from dataloader_preprocessed import data_generator_preprocessed
from models import get_model


class trainer(object):
    def __init__(self, args):
        # dataset parameters
        self.dataset = args.dataset
        self.seed_id = args.seed_id
        # 加载样本和标签
        self.device = torch.device(args.device)

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        # paths
        # External CV runs use a dataset/session-specific root so that runs
        # from different source datasets cannot overwrite one another.
        self.save_dir = getattr(args, "weights_root", "weights")
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.data_path = args.data_path
        self.exclude_channels = getattr(args, "exclude_channels", None)

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify hparams
        self.hparams = self.hparams_class.train_params

        self.models = args.models

    def _forward_model(self, model, data):
        outputs = model(data)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            aux_losses = [loss for loss in outputs[1:] if torch.is_tensor(loss)]
            aux_loss = torch.stack([loss.to(logits.device) for loss in aux_losses]).sum() if aux_losses else None
            return logits, aux_loss
        return outputs, None

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    # 训练集
    def load_data(self, data_type):
        self.train_dl, self.val_dl, self.test_dl = \
            data_generator_preprocessed(
                self.data_path, data_type, self.hparams,
                exclude_channels=self.exclude_channels,
            )
        train_dataset = self.train_dl.dataset
        self.dataset_configs.input_channels = train_dataset.raw_eeg_data.shape[1]
        self.dataset_configs.excluded_channels = list(train_dataset.excluded_channels)
        self.dataset_configs.original_ch_names = train_dataset.original_ch_names
        if getattr(train_dataset, "ch_names", None) is not None:
            self.dataset_configs.ch_names = list(train_dataset.ch_names)
            self.dataset_configs.input_channels = len(train_dataset.ch_names)
        if getattr(train_dataset, "sfreq", None) is not None:
            self.dataset_configs.sfreq = float(train_dataset.sfreq)
            self.dataset_configs.codimnet_mid_sfreq = float(train_dataset.sfreq) / 2.0
        message = (
            f"Loaded {self.dataset_configs.input_channels} channels: "
            f"{train_dataset.ch_names}; excluded: {train_dataset.excluded_channels}"
        )
        if getattr(self, "logger", None) is not None:
            self.logger.debug(message)
        else:
            print(message)

    # 使用 utils_twoclassify._calc_metrics（已更新为：micro Acc + macro 其余 + AUCPR + MCC，全部百分数）
    def calc_results_per_run(self):
        acc, f1, precision, recall, g_mean, auc_score, auc_pr, kappa, mcc = _calc_metrics(
            self.pred_labels, self.true_labels, self.pred_probs, self.dataset_configs.class_names
        )
        return acc, f1, precision, recall, g_mean, auc_score, auc_pr, kappa, mcc

    def train(self):
        # 训练过程中的累积记录（0~1 范围；日志统一使用 _calc_metrics 的百分数）
        self.metrics = {
            'accuracy': [], 'f1_score': [], 'precision': [], 'recall': [],
            'auc_score': [], 'aucpr': [], 'g_mean': [],
            'kappa': [], 'mcc': []
        }
        # fixing random seed
        fix_randomness(int(self.seed_id))

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.exp_log_dir, self.seed_id)
        self.logger.debug(self.hparams)
        self.tb_writer = None
        if SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.scenario_log_dir, "tensorboard"))
        else:
            self.logger.warning("TensorBoard is not available. Install tensorboard to enable SummaryWriter logs.")

        # 针对显存密集型模型提前减小 batch_size，只加载一次数据
        if self.models in ["EEGMambaClassifier", "CBraModClassifier"]:
            self.hparams["batch_size"] = 16

        # Load data
        self.load_data(self.dataset)

        # 模型选择
        model = self.select_model()

        # 在 CPU 上用第一个 batch 触发 lazy init（部分模型需要根据输入通道数构建网络）
        for batches in self.train_dl:
            dummy_data = batches['raw_eeg_data'].float()
            with torch.no_grad():
                _, _ = self._forward_model(model, dummy_data)
            break

        # 初始化完成后统一放到 GPU
        model.to(self.device)

        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            betas=(0.9, 0.99)
        )
        self.scheduler = None
        if self.hparams.get("lr_scheduler", "none") == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hparams.get("lr_step_size", 20),
                gamma=self.hparams.get("lr_gamma", 0.5),
            )
        self.cross_entropy = nn.CrossEntropyLoss()

        # Always save the first epoch, including when validation accuracy is 0.
        best_acc = float("-inf")
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        best_g_mean = 0
        best_auc_score = 0
        best_auc_pr = 0
        best_kappa = 0
        best_mcc = 0

        # training..
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            model.train()
            for meter in loss_avg_meters.values():
                meter.reset()

            for batches in self.train_dl:
                batches = to_device(batches, self.device)

                data = batches['raw_eeg_data'].float()
                labels = batches['labels'].long()

                # ====== forward ======
                self.optimizer.zero_grad()
                logits, aux_loss = self._forward_model(model, data)
                labels = labels.view(-1)  # 确保标签形状为 [batch_size,]

                # loss
                x_ent_loss = self.cross_entropy(logits, labels)
                total_loss = x_ent_loss
                if aux_loss is not None:
                    total_loss = total_loss + self.hparams.get("codimnet_pinn_weight", 0.0) * aux_loss

                total_loss.backward()
                self.optimizer.step()

                # logging meters
                losses = {
                    'Total_loss': total_loss.item(),
                    'CE_loss': x_ent_loss.item(),
                }
                if aux_loss is not None:
                    losses['Aux_loss'] = aux_loss.item()
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, self.hparams["batch_size"])

            # ======== EVAL on TRAIN ========
            self.evaluate_train(model, self.train_dl)
            tr_acc, tr_f1, tr_precision, tr_recall, tr_g_mean, tr_auc_score, tr_auc_pr, tr_kappa, tr_mcc = self.calc_results_per_run()

            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            self.logger.debug(f'Learning_rate\t: {self.optimizer.param_groups[0]["lr"]:2.6f}')
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
            self.logger.debug(
                f'TRAIN: Acc:{tr_acc:2.4f} \t F1:{tr_f1:2.4f} \t Precision:{tr_precision:2.4f} \t Recall:{tr_recall:2.4f} '
                f'\t G-Mean:{tr_g_mean:2.4f} \t AUC:{tr_auc_score:2.4f} '
                f'\t AUCPR:{tr_auc_pr:2.4f} \t Kappa:{tr_kappa:2.4f} \t MCC:{tr_mcc:2.4f}'
            )

            # ======== EVAL on VAL ========
            self.evaluate_train(model, self.val_dl)
            ts_acc, ts_f1, ts_precision, ts_recall, ts_g_mean, ts_auc_score, ts_auc_pr, ts_kappa, ts_mcc = self.calc_results_per_run()
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train", loss_avg_meters["Total_loss"].avg, epoch)
                self.tb_writer.add_scalar("train/accuracy", tr_acc, epoch)
                self.tb_writer.add_scalar("train/f1", tr_f1, epoch)
                self.tb_writer.add_scalar("train/precision", tr_precision, epoch)
                self.tb_writer.add_scalar("train/recall", tr_recall, epoch)
                self.tb_writer.add_scalar("train/auc", tr_auc_score, epoch)
                self.tb_writer.add_scalar("train/aucpr", tr_auc_pr, epoch)
                self.tb_writer.add_scalar("train/mcc", tr_mcc, epoch)
                self.tb_writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
                self.tb_writer.add_scalar("val/accuracy", ts_acc, epoch)
                self.tb_writer.add_scalar("val/f1", ts_f1, epoch)
                self.tb_writer.add_scalar("val/precision", ts_precision, epoch)
                self.tb_writer.add_scalar("val/recall", ts_recall, epoch)
                self.tb_writer.add_scalar("val/auc", ts_auc_score, epoch)
                self.tb_writer.add_scalar("val/aucpr", ts_auc_pr, epoch)
                self.tb_writer.add_scalar("val/mcc", ts_mcc, epoch)

            if ts_acc > best_acc:  # save best model by Acc
                best_f1 = ts_f1
                best_acc = ts_acc
                best_precision = ts_precision
                best_recall = ts_recall
                best_g_mean = ts_g_mean
                best_auc_score = ts_auc_score
                best_auc_pr = ts_auc_pr
                best_kappa = ts_kappa
                best_mcc = ts_mcc

                save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "best", self.models)
                _save_metrics(self.pred_labels, self.true_labels, self.pred_probs, self.exp_log_dir,
                              self.dataset_configs.class_names, "best", self.models)

            self.logger.debug(
                f'VAL(Patch):  Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f} \t Precision:{ts_precision:2.4f} \t Recall:{ts_recall:2.4f} '
                f'\t G-Mean:{ts_g_mean:2.4f} \t AUC:{ts_auc_score:2.4f} '
                f'\t AUCPR:{ts_auc_pr:2.4f} \t Kappa:{ts_kappa:2.4f} \t MCC:{ts_mcc:2.4f}'
            )

            self.logger.debug('-------------------------------------')
            if self.scheduler is not None:
                self.scheduler.step()

        # ======== LAST EPOCH 保存 ========
        _save_metrics(self.pred_labels, self.true_labels, self.pred_probs, self.exp_log_dir,
                      self.dataset_configs.class_names, "last", self.models)
        self.logger.debug("LAST EPOCH PERFORMANCE ...")
        self.logger.debug(
            f'Acc:{ts_acc:2.4f} \t F1:{ts_f1:2.4f} \t Precision:{ts_precision:2.4f} \t Recall:{ts_recall:2.4f} '
            f'\t G-Mean:{ts_g_mean:2.4f} \t AUC:{ts_auc_score:2.4f} '
            f'\t AUCPR:{ts_auc_pr:2.4f} \t Kappa:{ts_kappa:2.4f} \t MCC:{ts_mcc:2.4f}'
        )
        self.logger.debug(":::::::::::::")
        # ======== BEST EPOCH 汇报 ========
        self.logger.debug("BEST EPOCH PERFORMANCE ...")
        self.logger.debug(
            f'Acc:{best_acc:2.4f} \t F1:{best_f1:2.4f} \t Precision:{best_precision:2.4f} \t Recall:{best_recall:2.4f} '
            f'\t G-Mean:{best_g_mean:2.4f} \t AUC:{best_auc_score:2.4f} '
            f'\t AUCPR:{best_auc_pr:2.4f} \t Kappa:{best_kappa:2.4f} \t MCC:{best_mcc:2.4f}'
        )
        save_checkpoint(self.exp_log_dir, model, self.dataset, self.dataset_configs, self.hparams, "last", self.models)

        # ======== TESTING ========
        print(" === Evaluating on TEST set ===")
        best_model_path = os.path.join(self.exp_log_dir, f"best/checkpoint_{self.models}.pt")
        checkpoint = torch.load(best_model_path)  # Load the checkpoint
        model.load_state_dict(checkpoint['model'])  # Load the best model based on accuracy
        self.evaluate_train(model, self.test_dl)
        test_acc, test_f1, test_precision, test_recall, test_g_mean, test_auc_score, test_auc_pr, test_kappa, test_mcc = self.calc_results_per_run()

        _save_metrics(self.pred_labels, self.true_labels, self.pred_probs, self.exp_log_dir,
                      self.dataset_configs.class_names, "test", self.models)

        self.logger.debug("TEST PERFORMANCE (Patch-level):")
        self.logger.debug(
            f'Acc:{test_acc:2.4f} \t F1:{test_f1:2.4f} \t Precision:{test_precision:2.4f} \t Recall:{test_recall:2.4f} '
            f'\t G-Mean:{test_g_mean:2.4f} \t AUC:{test_auc_score:2.4f} '
            f'\t AUCPR:{test_auc_pr:2.4f} \t Kappa:{test_kappa:2.4f} \t MCC:{test_mcc:2.4f}'
        )
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("test/accuracy", test_acc, 0)
            self.tb_writer.add_scalar("test/f1", test_f1, 0)
            self.tb_writer.add_scalar("test/precision", test_precision, 0)
            self.tb_writer.add_scalar("test/recall", test_recall, 0)
            self.tb_writer.add_scalar("test/auc", test_auc_score, 0)
            self.tb_writer.add_scalar("test/aucpr", test_auc_pr, 0)
            self.tb_writer.add_scalar("test/mcc", test_mcc, 0)

        if self.tb_writer is not None:
            self.tb_writer.close()

    def select_model(self):
        return get_model(self.models, self.dataset_configs)

    def evaluate_train(self, model, dataset):
        """
        在给定数据迭代器上做一次普通的 batch 推断。
        计算规则：Accuracy 为微平均，其余（F1/Precision/Recall/G-Mean/AUC/AUCPR）均为宏平均（one-vs-rest 后求均值）。
        """
        model.to(self.device).eval()

        self.metrics = {
            'accuracy': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'auc_score': [],
            'aucpr': [],
            'g_mean': [],
            'kappa': [],
            'mcc': []
        }

        self.pred_labels = np.array([])
        self.true_labels = np.array([])
        self.pred_probs = np.array([])

        with torch.no_grad():
            for batches in dataset:
                batches = to_device(batches, self.device)
                data = batches['raw_eeg_data'].float()
                labels = batches['labels'].long()

                labels = labels.view(-1)
                predictions, _ = self._forward_model(model, data)

                probs = torch.softmax(predictions, dim=1).cpu().numpy()
                preds = predictions.detach().argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                if self.pred_probs.size == 0:
                    self.pred_probs = probs
                else:
                    self.pred_probs = np.vstack((self.pred_probs, probs))

                self.pred_labels = np.append(self.pred_labels, preds)
                self.true_labels = np.append(self.true_labels, labels_np)

        # ===== Metrics =====
        # Accuracy（micro）
        accuracy = accuracy_score(self.true_labels, self.pred_labels)

        # F1/Precision/Recall（macro）
        f1v = f1_score(self.true_labels, self.pred_labels, average='macro', zero_division=0)
        precisionv = precision_score(self.true_labels, self.pred_labels, average='macro', zero_division=0)
        recallv = recall_score(self.true_labels, self.pred_labels, average='macro', zero_division=0)

        # G-Mean（macro，逐类 one-vs-rest）
        class_ids = list(range(self.pred_probs.shape[1])) if self.pred_probs.ndim == 2 else list(np.unique(self.true_labels).astype(int))
        cm = confusion_matrix(self.true_labels, self.pred_labels, labels=class_ids)
        total = cm.sum()
        g_list = []
        for i in range(len(class_ids)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total - (tp + fn + fp)
            sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
            spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
            g_list.append(np.sqrt(sens_i * spec_i))
        g_mean = float(np.mean(g_list)) if len(g_list) else 0.0

        # AUC (macro) & AUCPR (macro) —— 逐类 one-vs-rest
        try:
            auc_vals, aupr_vals = [], []
            for i in range(len(class_ids)):
                class_true = (self.true_labels == class_ids[i]).astype(int)
                try:
                    auc_vals.append(roc_auc_score(class_true, self.pred_probs[:, i]))
                except Exception:
                    auc_vals.append(0.0)
                try:
                    aupr_vals.append(average_precision_score(class_true, self.pred_probs[:, i]))
                except Exception:
                    aupr_vals.append(0.0)
            aucv = float(np.mean(auc_vals)) if len(auc_vals) else 0.0
            aucpr = float(np.mean(aupr_vals)) if len(aupr_vals) else 0.0
        except Exception:
            aucv, aucpr = 0.0, 0.0

        # Kappa / MCC（全局）
        kappa = cohen_kappa_score(self.true_labels, self.pred_labels)
        try:
            mcc = matthews_corrcoef(self.true_labels, self.pred_labels)
        except Exception:
            mcc = 0.0

        # 累积（注意：这里是 0~1；日志里使用 _calc_metrics 的百分数版本）
        self.metrics['accuracy'].append(accuracy)
        self.metrics['f1_score'].append(f1v)
        self.metrics['precision'].append(precisionv)
        self.metrics['recall'].append(recallv)
        self.metrics['auc_score'].append(aucv)
        self.metrics['aucpr'].append(aucpr)
        self.metrics['g_mean'].append(g_mean)
        self.metrics['kappa'].append(kappa)
        self.metrics['mcc'].append(mcc)

        return self.metrics
