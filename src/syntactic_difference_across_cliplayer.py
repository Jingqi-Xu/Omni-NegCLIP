"""
计算 CLIP Text Encoder 每一层中，caption 和 negative caption 之间 EOS token 的余弦相似度。

这个脚本用于分析：在预训练 CLIP 模型的不同层中，正常描述和否定描述之间的语义差异如何变化。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import argparse

from data import FineTuningDataset
from configs import configs


def parse_args():
    parser = argparse.ArgumentParser(description="Compute cosine similarity of EOS tokens across CLIP layers")
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum samples to process (-1 for all)")
    return parser.parse_args()


def get_text_features_per_layer(model, text_tokens, device):
    """
    获取 CLIP text encoder 每一层的 EOS token 特征。
    
    CLIP Text Encoder 结构:
    1. token_embedding + positional_embedding
    2. transformer.resblocks[0-11]: 12 个 ResidualAttentionBlock
    3. ln_final
    4. text_projection (只在最后一层应用)
    
    Args:
        model: CLIP 模型
        text_tokens: tokenized text, shape [batch_size, context_length]
        device: 计算设备
        
    Returns:
        layer_features: dict, key 是层号 (0-11), value 是该层的 EOS token 特征
    """
    text_tokens = text_tokens.to(device)
    
    # 获取 EOS token 的位置 (argmax 找到最大值位置，即 EOT token)
    # 在 CLIP tokenization 中，EOT (end of text) token 的值最大
    eos_indices = text_tokens.argmax(dim=-1)  # [batch_size]
    
    # Step 1: Token embedding + positional embedding
    x = model.token_embedding(text_tokens).type(model.dtype)  # [batch_size, n_ctx, d_model]
    x = x + model.positional_embedding.type(model.dtype)
    
    # Step 2: 通过每一层 transformer block，收集每层的输出
    x = x.permute(1, 0, 2)  # [n_ctx, batch_size, d_model] - transformer 需要这个格式
    
    layer_features = {}
    
    for layer_idx, resblock in enumerate(model.transformer.resblocks):
        # 直接调用 resblock，attention mask 在 CLIP 内部通过 register_buffer 处理
        x = resblock(x)
        
        # 提取这一层的 EOS token 特征
        # x shape: [n_ctx, batch_size, d_model]
        x_permuted = x.permute(1, 0, 2)  # [batch_size, n_ctx, d_model]
        
        # 使用 gather 提取每个样本的 EOS token
        batch_size = x_permuted.shape[0]
        d_model = x_permuted.shape[2]
        
        # eos_indices: [batch_size] -> [batch_size, 1, d_model]
        eos_idx_expanded = eos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, d_model)
        eos_features = x_permuted.gather(1, eos_idx_expanded).squeeze(1)  # [batch_size, d_model]
        
        layer_features[layer_idx] = eos_features.float()  # 转为 float 以便后续计算
    
    return layer_features


def compute_cosine_similarity(features1, features2):
    """
    计算两组特征之间的余弦相似度。
    
    Args:
        features1: [batch_size, d_model]
        features2: [batch_size, d_model]
        
    Returns:
        similarities: [batch_size] 每对特征的余弦相似度
    """
    # 归一化
    features1_norm = F.normalize(features1, dim=-1)
    features2_norm = F.normalize(features2, dim=-1)
    
    # 计算余弦相似度 (对应元素相乘再求和)
    similarities = (features1_norm * features2_norm).sum(dim=-1)
    
    return similarities


@torch.no_grad()
def compute_layer_similarities(args):
    """
    计算整个数据集在每一层的 caption 和 negative caption 之间的平均余弦相似度。
    """
    print(f"Loading CLIP model: {args.clip_model_name}")
    model, preprocess = clip.load(args.clip_model_name, device=args.device)
    model.eval()
    
    print("Loading fine-tuning dataset...")
    dataset = FineTuningDataset(transform=preprocess)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    num_layers = len(model.transformer.resblocks)
    print(f"CLIP Text Encoder has {num_layers} transformer layers")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)
    
    # 用于累积每一层的相似度
    layer_similarity_sums = {i: 0.0 for i in range(num_layers)}
    total_samples = 0
    
    max_batches = args.max_samples // args.batch_size if args.max_samples > 0 else len(loader)
    
    pbar = tqdm(enumerate(loader), total=min(max_batches, len(loader)), desc="Computing similarities")
    
    for batch_idx, batch in pbar:
        if args.max_samples > 0 and batch_idx >= max_batches:
            break
            
        # batch: (images, captions, negative_captions)
        images, captions, negative_captions = batch
        
        # captions 和 negative_captions shape: [batch_size, 1, context_length]
        captions = captions.squeeze(1)  # [batch_size, context_length]
        negative_captions = negative_captions.squeeze(1)  # [batch_size, context_length]
        
        batch_size = captions.shape[0]
        total_samples += batch_size
        
        # 获取每一层的特征
        caption_layer_features = get_text_features_per_layer(model, captions, args.device)
        neg_caption_layer_features = get_text_features_per_layer(model, negative_captions, args.device)
        
        # 计算每一层的余弦相似度
        for layer_idx in range(num_layers):
            similarities = compute_cosine_similarity(
                caption_layer_features[layer_idx],
                neg_caption_layer_features[layer_idx]
            )
            layer_similarity_sums[layer_idx] += similarities.sum().item()
        
        # 更新进度条显示
        if (batch_idx + 1) % 10 == 0:
            avg_sim_layer0 = layer_similarity_sums[0] / total_samples
            avg_sim_layer11 = layer_similarity_sums[11] / total_samples
            pbar.set_postfix({
                'samples': total_samples,
                'L0_sim': f'{avg_sim_layer0:.4f}',
                'L11_sim': f'{avg_sim_layer11:.4f}'
            })
    
    # 计算每一层的平均相似度
    layer_avg_similarities = {i: layer_similarity_sums[i] / total_samples for i in range(num_layers)}
    
    return layer_avg_similarities, total_samples


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Syntactic Difference Analysis Across CLIP Layers")
    print("=" * 60)
    print(f"Model: {args.clip_model_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    print("=" * 60)
    print()
    
    layer_similarities, total_samples = compute_layer_similarities(args)
    
    print()
    print("=" * 60)
    print(f"Results (Total samples: {total_samples})")
    print("=" * 60)
    print()
    print("Average Cosine Similarity between Caption and Negative Caption")
    print("at EOS token position for each CLIP Text Encoder layer:")
    print()
    print(f"{'Layer':<10} {'Avg Cosine Similarity':<25} {'1 - Similarity (Difference)':<25}")
    print("-" * 60)
    
    for layer_idx in range(len(layer_similarities)):
        sim = layer_similarities[layer_idx]
        diff = 1 - sim
        print(f"Layer {layer_idx:<4} {sim:<25.6f} {diff:<25.6f}")
    
    print("-" * 60)
    print()
    
    # 打印一些总结性统计
    sims = list(layer_similarities.values())
    print(f"Minimum similarity: Layer {sims.index(min(sims))} = {min(sims):.6f}")
    print(f"Maximum similarity: Layer {sims.index(max(sims))} = {max(sims):.6f}")
    print(f"Similarity range: {max(sims) - min(sims):.6f}")
    print()
    
    # 分析趋势
    print("Layer group analysis:")
    front_avg = sum(sims[0:4]) / 4
    middle_avg = sum(sims[4:8]) / 4
    back_avg = sum(sims[8:12]) / 4
    print(f"  Front layers (0-3) average:  {front_avg:.6f}")
    print(f"  Middle layers (4-7) average: {middle_avg:.6f}")
    print(f"  Back layers (8-11) average:  {back_avg:.6f}")
    print()
    
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
