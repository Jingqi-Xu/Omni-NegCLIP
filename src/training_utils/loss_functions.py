import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
    
    def set_loss_type(self, new_loss_type):
        self.loss_type = new_loss_type 

    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels
    
    def forward(self, image_features, text_features, logit_scale):
        # image_features: [batch_size, dim]
        # text_features: [2*batch_size, dim] = [正确描述, 否定描述]
        batch_size = image_features.shape[0]
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)
        
        # 分离正确描述和否定描述（用于 L3）
        pos_text_features = text_features[:batch_size]   # [batch_size, dim]
        neg_text_features = text_features[batch_size:]   # [batch_size, dim]
        
        # L1: image → 所有文本（正确描述 + 否定描述）
        # 让 image_i 匹配 pos_text_i，同时远离所有其他文本（包括否定描述）
        logits_per_image = logit_scale * image_features @ text_features.T  # [batch_size, 2*batch_size]
        L1 = F.cross_entropy(logits_per_image, labels)
        
        # L2: 正确描述 → 所有图像 (让 pos_text_i 匹配 image_i)
        logits_per_text = logit_scale * text_features[:batch_size] @ image_features.T 
        L2 = F.cross_entropy(logits_per_text, labels)

        # L3: 对于每个图像，显式地在 [正确描述, 否定描述] 中选择正确描述
        # 这是对 L1 的补充，更直接地约束图像远离其对应的否定描述
        pos_sim = (image_features * pos_text_features).sum(dim=-1, keepdim=True)  # [batch_size, 1]
        neg_sim = (image_features * neg_text_features).sum(dim=-1, keepdim=True)  # [batch_size, 1]
        logits_for_negation = logit_scale * torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 2]
        labels_negation = torch.zeros(batch_size, dtype=torch.long, device=image_features.device)
        L3 = F.cross_entropy(logits_for_negation, labels_negation)
        #我们方法原本的loss in stage1
        total_loss = (L1 + L2 + L3) / 3
        #ablation for L3
        #total_loss = (L1 + L2) / 3
        
        return total_loss


class CustomLossWithNegatives(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
        self.loss_type = "negclip"
    
    def set_loss_type(self, new_loss_type):
        self.loss_type = new_loss_type
        
    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels

    def forward(self,
                image_features,
                text_features,
                logit_scale
                ):

        batch_size = image_features.shape[0] // 2
        # 0 -> batch_size -1 = positive
        # batch_size -> 2 * batch_size -1 = negative

        # get labels for loss calculation
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)

        text_features_rev = torch.cat([text_features[batch_size:], text_features[:batch_size]], dim=0)

        logits1 = logit_scale * image_features[:batch_size] @ text_features.T
        logits2 = logit_scale * text_features[:batch_size] @ image_features.T
        logits3 = logit_scale * image_features[batch_size:] @ text_features_rev.T

        total_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels) + F.cross_entropy(logits3, labels))/3

        return total_loss
        
class OriginalLoss(nn.Module):
    """
    与 CustomLoss 对比用的损失函数
    和 CustomLoss 一样使用完整的 text_features（包含 caption + negation caption）
    但是没有 L3（显式约束 image 远离 negation caption）
    
    L1: image → 所有文本（正确描述 + 否定描述）
    L2: 正确描述 → 所有图像
    （没有 L3）
    """
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
    
    def set_loss_type(self, new_loss_type):
        self.loss_type = new_loss_type 

    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels
    
    def forward(self, image_features, text_features, logit_scale):
        # image_features: [batch_size, dim]
        # text_features: [2*batch_size, dim] = [正确描述, 否定描述]
        batch_size = image_features.shape[0]
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)
        
        # L1: image → 所有文本（正确描述 + 否定描述）
        # 让 image_i 匹配 pos_text_i，同时远离所有其他文本（包括否定描述）
        logits_per_image = logit_scale * image_features @ text_features.T  # [batch_size, 2*batch_size]
        L1 = F.cross_entropy(logits_per_image, labels)
        
        # L2: 正确描述 → 所有图像 (让 pos_text_i 匹配 image_i)
        logits_per_text = logit_scale * text_features[:batch_size] @ image_features.T 
        L2 = F.cross_entropy(logits_per_text, labels)

        # 没有 L3！这是和 CustomLoss 的唯一区别
        total_loss = (L1 + L2) / 2
        
        return total_loss
        
class FinetuneclipLoss(nn.Module):
    """
    标准 CLIP 微调损失函数
    只使用正确描述，不使用否定描述
    
    L1: image → 正确描述
    L2: 正确描述 → image
    """
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
    
    def set_loss_type(self, new_loss_type):
        self.loss_type = new_loss_type 

    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels
    
    def forward(self, image_features, text_features, logit_scale):
        # image_features: [batch_size, dim]
        # text_features: [batch_size, dim] (只有正确描述，没有否定描述)
        batch_size = image_features.shape[0]
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)
        
        # L1: image → 正确描述
        logits_per_image = logit_scale * image_features @ text_features.T  # [batch_size, batch_size]
        L1 = F.cross_entropy(logits_per_image, labels)
        
        # L2: 正确描述 → image
        logits_per_text = logit_scale * text_features @ image_features.T  # [batch_size, batch_size]
        L2 = F.cross_entropy(logits_per_text, labels)

        total_loss = (L1 + L2) / 2
        
        return total_loss


def get_criterion(args):
    if args.negative_images == "off":
        return CustomLoss()
    elif args.negative_images == "on":
        return CustomLossWithNegatives()
    elif args.negative_images == "original":
        return OriginalLoss()
    elif args.negative_images == "finetuneclip":
        return FinetuneclipLoss()
