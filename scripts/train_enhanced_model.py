"""
增强型模型训练脚本

本脚本实现增强型模型的训练流程，支持监督学习和自定义损失权重。

运行方式:
    python scripts/train_enhanced_model.py --epochs 10 --batch-size 32

Example:
    python scripts/train_enhanced_model.py --epochs 50 --lr 1e-4 --save-path checkpoints/model_v1.pth
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

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


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device='cuda',
    save_path=None,
    log_interval=10
):
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 计算设备
        save_path: 模型保存路径
        log_interval: 日志打印间隔
    
    Returns:
        history: 训练历史
    """
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    print("=" * 70)
    print("增强型模型训练")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"批次大小: {train_loader.batch_size}")
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
            # 移动到设备
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # 训练步骤
            total_loss, losses = train_step(model, batch, optimizer)
            
            train_losses.append(total_loss)
            train_metrics['policy_loss'].append(losses['policy'])
            train_metrics['value_loss'].append(losses['value'])
            train_metrics['shanten_loss'].append(losses['shanten'])
            train_metrics['riichi_loss'].append(losses['riichi'])
            
            # 打印日志
            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss:.4f} "
                      f"(P: {losses['policy']:.4f}, "
                      f"V: {losses['value']:.4f}, "
                      f"S: {losses['shanten']:.4f}, "
                      f"R: {losses['riichi']:.4f})")
        
        # 计算训练集平均损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_metrics = {
            k: sum(v) / len(v) for k, v in train_metrics.items()
        }
        
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
        
        # 计算验证集平均损失
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_metrics = {
            k: sum(v) / len(v) for k, v in val_metrics.items()
        }
        
        # 记录历史
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['train_metrics'].append(avg_metrics)
        history['val_metrics'].append(avg_val_metrics)
        
        # 打印epoch结果
        print(f"\nEpoch [{epoch+1}/{num_epochs}] 完成:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"    Policy: {avg_metrics['policy_loss']:.4f}")
        print(f"    Value: {avg_metrics['value_loss']:.4f}")
        print(f"    Shanten: {avg_metrics['shanten_loss']:.4f}")
        print(f"    Riichi: {avg_metrics['riichi_loss']:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"    Policy: {avg_val_metrics['policy_loss']:.4f}")
        print(f"    Value: {avg_val_metrics['value_loss']:.4f}")
        print(f"    Shanten: {avg_val_metrics['shanten_loss']:.4f}")
        print(f"    Riichi: {avg_val_metrics['riichi_loss']:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_path:
                model.save(save_path)
                print(f"  ✓ 保存最佳模型到: {save_path}")
        
        print("-" * 70)
    
    print("=" * 70)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print("=" * 70)
    
    return history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练增强型模型')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/mjai_logs/',
                       help='数据目录路径')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='最大样本数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='权重衰减')
    
    # 模型参数
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=5,
                       help='残差层数')
    parser.add_argument('--num-actions', type=int, default=200,
                       help='动作数量')
    
    # 保存参数
    parser.add_argument('--save-path', type=str, default='checkpoints/model.pth',
                       help='模型保存路径')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='日志打印间隔')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cpu 或 cuda)')
    
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
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
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
    
    # 训练
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval
    )
    
    # 保存训练历史
    history_path = save_path.with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")


if __name__ == '__main__':
    main()
