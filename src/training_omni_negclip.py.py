

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import clip
from tqdm import tqdm
from training_utils.loss_functions import *
from training_utils.schedulers import *
from data import *
from training_utils.helpers import *
import argparse
import json
import os
from pathlib import Path
from PIL import Image
import math
from contextlib import suppress
import time


def get_gpu_memory_info(device):
    """
    获取 GPU 内存使用信息
    
    Returns:
        dict: 包含已分配内存、最大分配内存、缓存内存等信息 (单位: GB)
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    # 获取当前设备
    if isinstance(device, str):
        if device == "cuda":
            device_idx = 0
        elif device.startswith("cuda:"):
            device_idx = int(device.split(":")[1])
        else:
            return {"error": f"Invalid device: {device}"}
    else:
        device_idx = device
    
    # 获取内存信息 (bytes -> GB)
    allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device_idx) / (1024 ** 3)
    
    # 获取 GPU 总内存
    total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
    
    return {
        "allocated_gb": allocated,
        "max_allocated_gb": max_allocated,
        "reserved_gb": reserved,
        "max_reserved_gb": max_reserved,
        "total_gb": total_memory,
        "utilization_percent": (max_allocated / total_memory) * 100
    }


def format_time(seconds):
    """将秒数格式化为 时:分:秒 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# --- Layer Freezing Function ---
def freeze_text_encoder_layers(model, trainable_layers="front"):
    """
    Freeze specific layers of the text encoder.
    
    CLIP ViT-B/32 Text Encoder structure:
    - token_embedding: nn.Embedding
    - positional_embedding: nn.Parameter
    - transformer.resblocks[0-11]: 12 ResidualAttentionBlock layers
    - ln_final: LayerNorm
    - text_projection: nn.Parameter
    
    Args:
        model: CLIP model
        trainable_layers: which layers to keep trainable
            - "front": layers 0-3 (shallow layers)
            - "middle": layers 4-7 (middle layers)  
            - "back": layers 8-11 (deep layers)
            - "whole": layers 0-11 (all layers)
    
    Note: embeddings, ln_final, and text_projection are always trainable.
    """
    # Keep embeddings trainable
    model.token_embedding.weight.requires_grad = True
    model.positional_embedding.requires_grad = True
    
    # First, freeze all transformer blocks
    for block in model.transformer.resblocks:
        for param in block.parameters():
            param.requires_grad = False
    
    # Keep final layer norm and projection trainable
    for param in model.ln_final.parameters():
        param.requires_grad = True
    model.text_projection.requires_grad = True
    
    # Now unfreeze the specified transformer layers
    num_layers = len(model.transformer.resblocks)  # Should be 12 for ViT-B/32
    
    if trainable_layers == "front":
        # Unfreeze layers 0-3 (front/shallow layers)
        start_idx, end_idx = 0, 6
    elif trainable_layers == "middle":
        # Unfreeze layers 4-7 (middle layers)
        start_idx, end_idx = 4, 8
    elif trainable_layers == "back":
        # Unfreeze layers 8-11 (back/deep layers)
        start_idx, end_idx = 8, 12
    elif trainable_layers == "whole":
        # Unfreeze all layers (0-11)
        start_idx, end_idx = 0, 12
    else:
        raise ValueError(f"Unknown trainable_layers: {trainable_layers}")
    
    # Unfreeze the specified transformer layers
    for i in range(start_idx, end_idx):
        for param in model.transformer.resblocks[i].parameters():
            param.requires_grad = True
    
    # Keep logit_scale trainable
    model.logit_scale.requires_grad = True
    
    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    tqdm.write(f"\n=== Text Encoder Layer Freezing Summary ===")
    tqdm.write(f"Total transformer layers: {num_layers}")
    tqdm.write(f"Trainable transformer layers: {trainable_layers} (layers {start_idx}-{end_idx-1})")
    tqdm.write(f"Frozen transformer layers: {[i for i in range(num_layers) if i < start_idx or i >= end_idx]}")
    tqdm.write(f"Also trainable: embeddings, ln_final, text_projection, logit_scale")
    tqdm.write(f"---")
    tqdm.write(f"Total parameters: {total_params:,}")
    tqdm.write(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    tqdm.write(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    tqdm.write("=" * 45 + "\n")
    
    return model


# --- Stage2 Dataset Definition ---
class COCOCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset for COCO captions (Stage2).
    返回 (image, original_caption, updated_caption)
    """
    def __init__(self, captions_file: Path, image_dir: Path, preprocess: callable):
        super().__init__()
        with open(captions_file, 'r') as f:
            coco_data = json.load(f)
        self.annotations = coco_data['annotations']
        self.images = {img['id']: img['file_name'] for img in coco_data['images']}
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        image_file_name = self.images[image_id]
        image_path = self.image_dir / image_file_name

        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)

        original_caption = annotation['caption']
        updated_caption = annotation['updated_caption']

        return processed_image, original_caption, updated_caption


# --- Loss Functions ---
def symmetric_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Calculates the symmetric contrastive loss for Stage2."""
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2.0


def text_contrastive_loss(original_features: torch.Tensor, 
                          updated_features: torch.Tensor, 
                          margin: float = 0.2) -> torch.Tensor:
    """
    Text Contrastive Loss: 让 original caption 和 updated caption 在特征空间中分开。
    
    目标：降低同一样本的 original 和 updated caption 之间的相似度。
    
    Args:
        original_features: 原始描述的特征 [batch_size, embedding_dim]，已归一化
        updated_features: 更新描述的特征 [batch_size, embedding_dim]，已归一化
        margin: 间隔阈值，目标是让相似度低于 -margin
        
    Returns:
        loss: 标量 loss
        
    原理：
        - 计算同一样本的 original 和 updated 之间的余弦相似度
        - 使用 Hinge Loss：如果相似度 > -margin，则有惩罚
        - 目标：让 "a dog" 和 "not a dog" 的相似度低于 -margin
    """
    # 计算同一样本的 original 和 updated 之间的余弦相似度
    # 因为特征已归一化，点积就是余弦相似度
    diagonal_sim = (original_features * updated_features).sum(dim=-1)  # [batch_size]
    
    # Hinge Loss: 希望 diagonal_sim < -margin
    loss = torch.clamp(diagonal_sim + margin, min=0).mean()
    
    return loss


def compute_stage2_loss_with_text_contrastive(model, images, original_captions, updated_captions, 
                                               device, lambda_orig=1.0, lambda_text=0.3, margin=0.2):
    """
    计算第二阶段的 loss（包含 Text Contrastive Loss）
    
    Loss = loss_negation + lambda_orig * loss_original + lambda_text * loss_text_contrastive
    
    - loss_negation: image ↔ updated_caption（拉近）
    - loss_original: image ↔ original_caption（保持对齐）
    - loss_text_contrastive: original_caption vs updated_caption（在文本空间中分开）
    """
    images = images.to(device)
    
    # Tokenize both captions
    updated_text_inputs = clip.tokenize(updated_captions, truncate=True).to(device)
    original_text_inputs = clip.tokenize(original_captions, truncate=True).to(device)

    # Forward pass - encode image once
    image_features = model.encode_image(images)
    updated_text_features = model.encode_text(updated_text_inputs)
    original_text_features = model.encode_text(original_text_inputs)

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    updated_text_features = updated_text_features / updated_text_features.norm(dim=-1, keepdim=True)
    original_text_features = original_text_features / original_text_features.norm(dim=-1, keepdim=True)

    # Calculate logits
    logit_scale = model.logit_scale.exp()
    
    # image ↔ updated_caption
    logits_per_image_updated = logit_scale * image_features @ updated_text_features.t()
    logits_per_text_updated = logits_per_image_updated.t()
    
    # image ↔ original_caption
    logits_per_image_original = logit_scale * image_features @ original_text_features.t()
    logits_per_text_original = logits_per_image_original.t()

    # Calculate losses
    # L1: image ↔ updated_caption (negation)
    loss_negation = symmetric_contrastive_loss(logits_per_image_updated, logits_per_text_updated)
    
    # L2: image ↔ original_caption
    loss_original = symmetric_contrastive_loss(logits_per_image_original, logits_per_text_original)
    
    # L3: text contrastive loss (original vs updated)
    loss_text = text_contrastive_loss(original_text_features, updated_text_features, margin=margin)
    
    # Combined loss
    total_loss = loss_negation + lambda_orig * loss_original + lambda_text * loss_text
    
    return total_loss, loss_negation.item(), loss_original.item(), loss_text.item()


def train_one_epoch_mix(args, epoch, model, loader_stage1, loader_stage2, 
                        optimizer, criterion_stage1, scheduler, scaler):
    """
    联合训练一个 epoch
    每个 step 同时使用两个阶段的数据
    """
    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress

    # 记录 epoch 开始时间
    epoch_start_time = time.time()
    
    # 创建迭代器
    iter_stage1 = iter(loader_stage1)
    iter_stage2 = iter(loader_stage2)
    
    # 总步数取两个 loader 的最大值
    total_steps = max(len(loader_stage1), len(loader_stage2))
    
    bar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}")
    running_loss = 0
    running_loss_s1 = 0
    running_loss_s2 = 0
    running_loss_s2_neg = 0
    running_loss_s2_orig = 0
    running_loss_s2_text = 0
    
    for i in range(total_steps):
        step = total_steps * epoch + i
        scheduler(step)
        
        # ============ 获取 Stage1 数据 ============
        try:
            batch_s1 = next(iter_stage1)
        except StopIteration:
            iter_stage1 = iter(loader_stage1)
            batch_s1 = next(iter_stage1)
        
        # 解析 Stage1 batch
        negative_images = None
        if len(batch_s1) == 4:
            images_s1, negative_images, captions_s1, negative_captions_s1 = batch_s1
        elif len(batch_s1) == 3:
            images_s1, captions_s1, negative_captions_s1 = batch_s1
        
        # 准备 Stage1 文本
        all_texts_s1 = torch.cat([captions_s1, negative_captions_s1], dim=0).squeeze(1)
        all_texts_s1 = all_texts_s1.to(args.device, non_blocking=True)
        
        if negative_images is not None:
            all_images_s1 = torch.cat([images_s1, negative_images], dim=0).to(args.device, non_blocking=True)
        else:
            all_images_s1 = images_s1.to(args.device, non_blocking=True)
        
        # ============ 获取 Stage2 数据 ============
        try:
            batch_s2 = next(iter_stage2)
        except StopIteration:
            iter_stage2 = iter(loader_stage2)
            batch_s2 = next(iter_stage2)
        
        images_s2, original_captions_s2, updated_captions_s2 = batch_s2
        
        # ============ 计算 Loss ============
        optimizer.zero_grad()
        
        with autocast():
            # Stage1 loss
            image_features_s1, text_features_s1, logit_scale = clip_forward_pass(model, all_images_s1, all_texts_s1)
            loss_stage1 = criterion_stage1(image_features_s1, text_features_s1, logit_scale)
            
            # Stage2 loss (with text contrastive loss)
            loss_stage2, loss_s2_neg, loss_s2_orig, loss_s2_text = compute_stage2_loss_with_text_contrastive(
                model, images_s2, original_captions_s2, updated_captions_s2, 
                args.device, 
                lambda_orig=args.lambda_orig,
                lambda_text=args.lambda_text,
                margin=args.margin
            )
            
            # 联合 loss
            total_loss = loss_stage1 + args.lambda_stage2 * loss_stage2
            
            running_loss += total_loss.item()
            running_loss_s1 += loss_stage1.item()
            running_loss_s2 += loss_stage2.item()
            running_loss_s2_neg += loss_s2_neg
            running_loss_s2_orig += loss_s2_orig
            running_loss_s2_text += loss_s2_text

        # ============ 反向传播 ============
        if scaler is not None:
            scaler.scale(total_loss).backward()

            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            optimizer.step()

        # clamp logit_scale
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100)) 

        # 更新进度条
        avg_loss = running_loss / (i+1)
        avg_loss_s1 = running_loss_s1 / (i+1)
        avg_loss_s2 = running_loss_s2 / (i+1)
        avg_loss_s2_text = running_loss_s2_text / (i+1)
        logs = {
            "loss": f"{avg_loss:.4f}",
            "L_s1": f"{avg_loss_s1:.4f}",
            "L_s2": f"{avg_loss_s2:.4f}",
            "L_txt": f"{avg_loss_s2_text:.4f}"
        }
        bar.update(1)
        bar.set_postfix(logs)
    
    bar.close()
    
    # 计算 epoch 训练时间
    epoch_time = time.time() - epoch_start_time
    
    return {
        "avg_loss": avg_loss, 
        "avg_loss_s1": avg_loss_s1, 
        "avg_loss_s2": avg_loss_s2,
        "avg_loss_s2_neg": running_loss_s2_neg / total_steps,
        "avg_loss_s2_orig": running_loss_s2_orig / total_steps,
        "avg_loss_s2_text": running_loss_s2_text / total_steps,
        "epoch_time_seconds": epoch_time
    }


def train_mix(args):
    """联合训练主函数"""
    tqdm.write("=" * 60)
    tqdm.write("Joint Training (Stage1 + Stage2) + Text Contrastive Loss")
    tqdm.write(f"Trainable layers: {args.trainable_layers}")
    tqdm.write("=" * 60)
    
    # 记录总训练开始时间
    total_start_time = time.time()
    
    # 重置 GPU 内存统计（用于准确跟踪最大内存使用）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # ============ 加载模型 ============
    model, preprocess = clip.load(args.clip_model_name, device=args.device)
    
    if args.precision == "amp":
        model = model.float()
        model = model.to(args.device)

    # 锁定视觉编码器
    if args.lock_image_encoder == "on":
        for param in model.visual.parameters():
            param.requires_grad = False
        tqdm.write("Locked the visual encoder.")

    # === Apply layer-wise freezing to text encoder ===
    model = freeze_text_encoder_layers(model, trainable_layers=args.trainable_layers)

    model.train()
    tqdm.write(f"CLIP: {args.clip_model_name} loaded and set to train")
    
    # ============ 加载 Stage1 数据集（CC-Neg）============
    dataset_stage1 = get_finetuning_dataset(args, preprocess)
    loader_stage1 = DataLoader(
        dataset_stage1, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True,
        shuffle=True
    )
    tqdm.write(f"Stage1 dataset loaded: {len(dataset_stage1)} samples")
    
    # ============ 加载 Stage2 数据集（COCO）============
    captions_file = Path(args.json_path)
    image_dir = Path(args.image_dir)
    dataset_stage2 = COCOCaptionDataset(captions_file, image_dir, preprocess)
    loader_stage2 = DataLoader(
        dataset_stage2, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True,
        shuffle=True
    )
    tqdm.write(f"Stage2 dataset loaded: {len(dataset_stage2)} samples")
    
    # ============ 设置训练参数 ============
    total_steps_per_epoch = max(len(loader_stage1), len(loader_stage2))
    total_steps = args.epochs * total_steps_per_epoch
    
    criterion_stage1 = get_criterion(args)
    
    # 只优化需要梯度的参数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    tqdm.write(f"Total steps per epoch: {total_steps_per_epoch}")
    tqdm.write(f"Total training steps: {total_steps}")
    tqdm.write("")
    tqdm.write("=== Loss Configuration ===")
    tqdm.write(f"Total Loss = L_stage1 + {args.lambda_stage2} * L_stage2")
    tqdm.write(f"L_stage2 = L_neg + {args.lambda_orig} * L_orig + {args.lambda_text} * L_text_contrastive")
    tqdm.write(f"Text Contrastive Loss margin: {args.margin}")
    tqdm.write("=" * 30)
    tqdm.write("")
    tqdm.write("Here we go!")
    tqdm.write("")

    logs = {"fine_tuning": {}}

    # ============ 训练循环 ============
    for epoch in range(args.epochs):
        training_logs = train_one_epoch_mix(
            args, epoch, model, loader_stage1, loader_stage2,
            optimizer, criterion_stage1, scheduler, scaler
        )
        
        logs["fine_tuning"][f"epoch_{epoch+1}"] = training_logs
        
        # 获取 GPU 内存信息
        gpu_info = get_gpu_memory_info(args.device)
        epoch_time = training_logs.get("epoch_time_seconds", 0)
        
        # 打印 epoch 统计信息
        tqdm.write("")
        tqdm.write(f"=== Epoch {epoch+1}/{args.epochs} Summary ===")
        tqdm.write(f"Epoch Time: {format_time(epoch_time)}")
        if "error" not in gpu_info:
            tqdm.write(f"GPU Memory - Current: {gpu_info['allocated_gb']:.2f} GB, "
                      f"Peak: {gpu_info['max_allocated_gb']:.2f} GB, "
                      f"Total: {gpu_info['total_gb']:.2f} GB "
                      f"({gpu_info['utilization_percent']:.1f}% utilized)")
        tqdm.write(f"Loss - Total: {training_logs['avg_loss']:.4f}, "
                  f"Stage1: {training_logs['avg_loss_s1']:.4f}, "
                  f"Stage2: {training_logs['avg_loss_s2']:.4f}")
        tqdm.write("=" * 40)
        tqdm.write("")
        
        # 保存 checkpoint
        check_and_save_mix(args, model, optimizer, logs, epoch)

    # 最终保存
    save_at_end_mix(args, model, optimizer, logs, epoch)
    
    # 打印总训练统计
    total_training_time = time.time() - total_start_time
    final_gpu_info = get_gpu_memory_info(args.device)
    
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write("Training Completed!")
    tqdm.write("=" * 60)
    tqdm.write(f"Total Training Time: {format_time(total_training_time)}")
    tqdm.write(f"Average Time per Epoch: {format_time(total_training_time / args.epochs)}")
    if "error" not in final_gpu_info:
        tqdm.write(f"Peak GPU Memory Used: {final_gpu_info['max_allocated_gb']:.2f} GB "
                  f"/ {final_gpu_info['total_gb']:.2f} GB "
                  f"({final_gpu_info['utilization_percent']:.1f}%)")
    tqdm.write("=" * 60)
    tqdm.write("")
    tqdm.write("Done!")


def check_and_save_mix(args, model, optimizer, logs, epoch):
    """保存 checkpoint"""
    if (epoch + 1) % args.save_every == 0:
        dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
        
        save_name = f"mix_textloss_{args.trainable_layers}_checkpoint_{epoch+1}_{args.experiment_name}.pt"
        save_folder = os.path.join(args.ckpt_save_folder, args.experiment_name)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, save_name)
        torch.save(dump, save_path)

        print(f"Checkpoint saved at epoch: {epoch+1}!")


def save_at_end_mix(args, model, optimizer, logs, epoch):
    """训练结束时保存"""
    save_name = f"mix_textloss_{args.trainable_layers}_checkpoint_{epoch+1}_{args.experiment_name}.pt"
    save_folder = os.path.join(args.ckpt_save_folder, args.experiment_name)
    os.makedirs(save_folder, exist_ok=True)
    
    save_path = os.path.join(save_folder, save_name)
    dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
    torch.save(dump, save_path)

    logs_save_path = os.path.join(
        args.logs_save_folder,
        args.experiment_name,
        f"results_{args.experiment_name}.pt"
    )
    os.makedirs(os.path.join(args.logs_save_folder, args.experiment_name), exist_ok=True)
    torch.save(logs, logs_save_path)


def setup_args_mix():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Joint Training for CoN-CLIP with Text Contrastive Loss")
    
    # Model args
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lock-image-encoder", type=str, default="on")
    
    # Layer-wise training
    parser.add_argument("--trainable-layers", type=str, default="front", 
                        choices=["front", "middle", "back", "whole"],
                        help="Which transformer layers to train: front (0-3), middle (4-7), back (8-11), or whole (0-11).")

    # Training args
    parser.add_argument("--learning-rate", type=float, default=1e-6) 
    parser.add_argument("--precision", type=str, default="amp") 
    parser.add_argument("--norm-gradient-clip", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.2)

    # Data args - Stage1 (CC-Neg)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--negative-images", type=str, default="off")  # 使用 CustomLoss
    
    # Data args - Stage2 (COCO)
    parser.add_argument("--json-path", type=str, required=True, help="Path to COCO captions JSON")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to COCO images directory")
    
    # Loss weights
    parser.add_argument("--lambda-stage2", type=float, default=1.0, help="Weight for Stage2 loss")
    parser.add_argument("--lambda-orig", type=float, default=1.0, help="Weight for original caption loss in Stage2")
    parser.add_argument("--lambda-text", type=float, default=1.0, help="Weight for text contrastive loss in Stage2")
    parser.add_argument("--margin", type=float, default=0.9, help="Margin for text contrastive loss")

    # Save args
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--ckpt-save-folder", type=str, default="../checkpoints_mix_textloss")
    parser.add_argument("--logs-save-folder", type=str, default="../logs_mix_textloss")
    parser.add_argument("--experiment-name", type=str, default="conclip_mix_textloss")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = setup_args_mix()
    train_mix(args)

