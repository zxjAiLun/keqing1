"""
增强型模型训练脚本 V2

本脚本在V1基础上增加了：
- 学习率调度器
- 早停法
- 模型集成
- 更好的日志记录

运行方式:
    python scripts/train_enhanced_model_v2.py --epochs 100 --batch-size 32
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import DataLoader

from v3model import (
    EnhancedModel,
    create_optimizer,
    train_step,
    validate
)
from v3model import (
    EnhancedDataset,
    create_dataloader
)


class EarlyStopping:
    """
    早停法
    
    当验证损失不再下降时停止训练
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 容忍次数
            min_delta: 最小改善
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        Returns:
            True if should early stop
        """
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class ModelEnsemble:
    """
    模型集成
    
    集成多个模型进行预测
    """
    
    def __init__(self, models: List[EnhancedModel]):
        self.models = models
    
    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        集成预测
        
        Args:
            features: 输入特征
        
        Returns:
            各模型输出的平均值
        """
        outputs_list = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(features, return_all_heads=True)
                outputs_list.append(outputs)
        
        # 平均各输出
        ensemble_outputs = {}
        for key in outputs_list[0].keys():
            if isinstance(outputs_list[0][key], torch.Tensor):
                values = torch.stack([o[key] for o in outputs_list])
                ensemble_outputs[key] = values.mean(dim=0)
            else:
                ensemble_outputs[key] = outputs_list[0][key]
        
        return ensemble_outputs
    
    def add_model(self, model: EnhancedModel):
        """添加模型到集成"""
        self.models.append(model)
    
    def save(self, path: str | Path):
        """保存集成模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = path.parent / f"{path.stem}_model_{i}.pth"
            model.save(model_path)
            model_paths.append(str(model_path))
        
        # 保存集成元数据
        metadata = {
            'num_models': len(self.models),
            'model_paths': model_paths,
            'version': '1.0.0'
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> 'ModelEnsemble':
        """加载集成模型"""
        path = Path(path)
        
        with open(path.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)
        
        models = []
        for model_path in metadata['model_paths']:
            models.append(EnhancedModel.load(model_path))
        
        return cls(models)


def train_with_advanced_features(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device='cpu',
    save_path=None,
    log_interval=10,
    scheduler_type='step',
    scheduler_params=None,
    early_stopping=None,
    ensemble_models=None
):
    """
    高级训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据
        val_loader: 验证数据
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
        save_path: 保存路径
        log_interval: 日志间隔
        scheduler_type: 调度器类型 ('step', 'cosine', 'plateau')
        scheduler_params: 调度器参数
        early_stopping: 早停对象
        ensemble_models: 集成模型列表
    
    Returns:
        history: 训练历史
    """
    # 学习率调度器
    scheduler_params = scheduler_params or {}
    
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.5)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', num_epochs)
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 5)
        )
    else:
        scheduler = None
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_metrics': [],
        'val_metrics': [],
        'learning_rates': [],
        'best_val_loss': float('inf')
    }
    
    print("=" * 70)
    print("增强型模型训练 V2")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率调度: {scheduler_type}")
    if early_stopping:
        print(f"早停法: patience={early_stopping.patience}")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'shanten_loss': [],
            'riichi_loss': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            total_loss, losses = train_step(model, batch, optimizer)
            
            train_losses.append(total_loss)
            train_metrics['policy_loss'].append(losses['policy'])
            train_metrics['value_loss'].append(losses['value'])
            train_metrics['shanten_loss'].append(losses['shanten'])
            train_metrics['riichi_loss'].append(losses['riichi'])
            
            if batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss:.4f} "
                      f"LR: {current_lr:.6f}")
        
        # 学习率调度
        if scheduler_type == 'plateau':
            avg_val_loss = sum(train_losses) / len(train_losses)
            scheduler.step(avg_val_loss)
        elif scheduler:
            scheduler.step()
        
        # 计算训练集平均
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_metrics = {k: sum(v) / len(v) for k, v in train_metrics.items()}
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'shanten_loss': [],
            'riichi_loss': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                total_loss, losses = validate(model, batch)
                
                val_losses.append(total_loss)
                val_metrics['policy_loss'].append(losses['policy'])
                val_metrics['value_loss'].append(losses['value'])
                val_metrics['shanten_loss'].append(losses['shanten'])
                val_metrics['riichi_loss'].append(losses['riichi'])
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}
        
        # 记录历史
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_metrics'].append(avg_metrics)
        history['val_metrics'].append(avg_val_metrics)
        history['learning_rates'].append(current_lr)
        
        # 打印结果
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 完成:")
        print(f"  训练损失: {avg_train_loss:.4f} "
              f"(P: {avg_metrics['policy_loss']:.4f}, "
              f"V: {avg_metrics['value_loss']:.4f}, "
              f"S: {avg_metrics['shanten_loss']:.4f}, "
              f"R: {avg_metrics['riichi_loss']:.4f})")
        print(f"  验证损失: {avg_val_loss:.4f} "
              f"(P: {avg_val_metrics['policy_loss']:.4f}, "
              f"V: {avg_val_metrics['value_loss']:.4f}, "
              f"S: {avg_val_metrics['shanten_loss']:.4f}, "
              f"R: {avg_val_metrics['riichi_loss']:.4f})")
        print(f"  学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_val_loss'] = best_val_loss
            if save_path:
                model.save(save_path)
                print(f"  ✓ 保存最佳模型到: {save_path}")
        
        # 早停法检查
        if early_stopping and early_stopping(avg_val_loss):
            print(f"\n早停！验证损失连续{early_stopping.patience}轮未改善")
            break
        
        print("-" * 70)
    
    print("=" * 70)
    print("训练完成！")
    print(f"最佳验证损失: {history['best_val_loss']:.4f}")
    print("=" * 70)
    
    return history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强型模型训练 V2')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/mjai_logs/')
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    
    # 学习率调度器
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine', 'plateau'],
                       help='学习率调度器类型')
    parser.add_argument('--step-size', type=int, default=10,
                       help='StepLR的step_size')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='学习率衰减因子')
    
    # 早停法
    parser.add_argument('--early-stopping', action='store_true',
                       help='启用早停法')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')
    
    # 模型参数
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--num-actions', type=int, default=200)
    
    # 保存参数
    parser.add_argument('--save-path', type=str, default='checkpoints/model_v2.pth')
    parser.add_argument('--init-checkpoint', type=str, default=None,
                       help='初始化模型权重路径')
    parser.add_argument('--log-interval', type=int, default=10)
    
    # 设备
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    dataset = EnhancedDataset(
        data_dir=args.data_dir,
        max_samples=args.max_samples
    )
    
    # 划分数据集
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 初始化模型
    print("初始化模型...")
    if args.init_checkpoint:
        print(f"从检查点加载模型: {args.init_checkpoint}")
        model = EnhancedModel.load(args.init_checkpoint)
        print("模型加载成功!")
    else:
        model = EnhancedModel(config={
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_actions': args.num_actions
        })
    model.to(args.device)
    
    # 创建优化器
    optimizer = create_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 早停法
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)
    
    # 调度器参数
    scheduler_params = {
        'step_size': args.step_size,
        'gamma': args.gamma
    }
    
    # 训练
    history = train_with_advanced_features(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval,
        scheduler_type=args.scheduler,
        scheduler_params=scheduler_params,
        early_stopping=early_stopping
    )
    
    # 保存历史
    history_path = save_path.with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")


if __name__ == '__main__':
    main()
