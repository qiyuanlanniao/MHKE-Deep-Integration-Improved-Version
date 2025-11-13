import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, ChineseCLIPModel


# ==============================================================================
# 1. NEW: 跨模态融合模块 (核心改进)
# ==============================================================================
class CrossModalFusionBlock(nn.Module):
    """
    一个基于交叉注意力的跨模态融合模块。
    它接收文本和图像的序列特征，并进行双向信息交互。
    """

    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        # 文本特征“查询”图像特征
        self.txt_to_img_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 图像特征“查询”文本特征
        self.img_to_txt_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 一个标准的前馈网络 (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, text_features, image_features):
        """
        Args:
            text_features (torch.Tensor): 文本序列特征, 形状 (batch_size, seq_len_txt, hidden_dim)
            image_features (torch.Tensor): 图像序列特征, 形状 (batch_size, seq_len_img, hidden_dim)
        Returns:
            torch.Tensor: 融合后的联合表征, 形状 (batch_size, hidden_dim)
        """
        # 1. 文本查询图像 (Text queries Image)
        attended_img, _ = self.txt_to_img_attention(text_features, image_features, image_features)
        fused_text = self.layer_norm1(text_features + attended_img)

        # 2. 图像查询文本 (Image queries Text)
        attended_txt, _ = self.img_to_txt_attention(image_features, text_features, text_features)
        fused_image = self.layer_norm2(image_features + attended_txt)

        # 3. 池化 (Pooling)
        text_cls = fused_text[:, 0, :]
        image_cls = fused_image[:, 0, :]
        pooled_representation = torch.mean(torch.stack([text_cls, image_cls]), dim=0)

        # 4. 通过FFN层
        output = self.layer_norm_ffn(pooled_representation + self.ffn(pooled_representation))

        return output


# ==============================================================================
# 2. MODIFIED: 改进后的 MHKE 模型 (基于 RoBERTa + ViT)
# ==============================================================================
class MHKE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.weight = config.weight

        print("Loading ViT and RoBERTa models for MHKE...")
        self.cv_model = ViTModel.from_pretrained(config.vit_path)
        self.nlp_model = BertModel.from_pretrained(config.roberta_path)

        self.cross_modal_fusion_layer = CrossModalFusionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.fusion_num_heads,
            dropout_rate=config.fusion_dropout
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, **args):
        text_outputs = self.nlp_model(
            input_ids=args['input_ids'].to(self.device),
            attention_mask=args['attention_mask'].to(self.device)
        )
        text_discription_outputs = self.nlp_model(
            input_ids=args['text_discription_input_ids'].to(self.device),
            attention_mask=args['text_discription_attention_mask'].to(self.device)
        )
        meme_discription_outputs = self.nlp_model(
            input_ids=args['meme_discription_input_ids'].to(self.device),
            attention_mask=args['meme_discription_attention_mask'].to(self.device)
        )
        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device)
        )

        text_sequence = text_outputs.last_hidden_state
        text_discription_sequence = text_discription_outputs.last_hidden_state
        meme_discription_sequence = meme_discription_outputs.last_hidden_state
        image_sequence = image_outputs.last_hidden_state

        knowledge_enhanced_text_sequence = (
                text_sequence +
                self.weight * text_discription_sequence +
                self.weight * meme_discription_sequence
        )

        fused_features = self.cross_modal_fusion_layer(
            text_features=knowledge_enhanced_text_sequence,
            image_features=image_sequence
        )
        output = self.classifier(fused_features)

        return output


# ==============================================================================
# 3. MODIFIED: 改进后的 MHKE_CLIP 模型
# ==============================================================================
class MHKE_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.weight = config.weight

        print("Loading ChineseCLIPModel for MHKE_CLIP...")
        self.model = ChineseCLIPModel.from_pretrained(config.clip_path)

        self.cross_modal_fusion_layer = CrossModalFusionBlock(
            hidden_dim=config.hidden_dim,
            num_heads=config.fusion_num_heads,
            dropout_rate=config.fusion_dropout
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, **args):
        text_discription_outputs = self.model.text_model(
            input_ids=args['text_discription_input_ids'].to(self.device),
            attention_mask=args['text_discription_attention_mask'].to(self.device)
        )
        meme_discription_outputs = self.model.text_model(
            input_ids=args['meme_discription_input_ids'].to(self.device),
            attention_mask=args['meme_discription_attention_mask'].to(self.device)
        )
        text_outputs = self.model.text_model(
            input_ids=args['input_ids'].to(self.device),
            attention_mask=args['attention_mask'].to(self.device)
        )
        image_outputs = self.model.vision_model(
            pixel_values=args['image_tensor'].to(self.device)
        )

        text_sequence = text_outputs.last_hidden_state
        text_discription_sequence = text_discription_outputs.last_hidden_state
        meme_discription_sequence = meme_discription_outputs.last_hidden_state
        image_sequence = image_outputs.last_hidden_state

        knowledge_enhanced_text = (
                text_sequence +
                self.weight * text_discription_sequence +
                self.weight * meme_discription_sequence
        )

        fused_features = self.cross_modal_fusion_layer(
            text_features=knowledge_enhanced_text,
            image_features=image_sequence
        )

        output = self.classifier(fused_features)
        return output
