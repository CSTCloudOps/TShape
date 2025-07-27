
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScalePatchConvolution(nn.Module):

    def __init__(self, patch_size, kernel_sizes=[2,3], out_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # 每个卷积核的输出通道数（均等分配）
        self.channels_per_scale = out_channels // len(kernel_sizes)
        self.total_channels = self.channels_per_scale * len(kernel_sizes)
        
        # 并行多尺度卷积层
        self.conv_layers = nn.ModuleList()
        for ks in kernel_sizes:
            conv = nn.Conv1d(
                in_channels=1, 
                out_channels=self.channels_per_scale, 
                kernel_size=ks,
                padding=ks//2  # 保持时间维度长度
            )
            self.conv_layers.append(conv)
        
        # 批归一化
        self.norm = nn.BatchNorm1d(self.total_channels)
        self.activation = nn.GELU()

    def forward(self, x):

        batch_size, seq_len = x.shape
        num_patches = seq_len // self.patch_size
        
        # 1. 分块: 将序列分割为独立的小块
        # 形状: (B, num_patches, patch_size)
        patches = x.view(batch_size, num_patches, self.patch_size)
        
        # 处理每个patch并应用多尺度卷积
        all_features = []
        
        # 迭代每个patch
        for i in range(num_patches):
            patch = patches[:, i, :]  # (B, patch_size)
            
            # 添加通道维度: (B, 1, patch_size)
            patch = patch.unsqueeze(1)
            
            # 应用所有卷积核
            patch_features = []
            for conv in self.conv_layers:
                conv_out = conv(patch)  # (B, C_per_scale, patch_size)
                
                # 沿时间维度全局平均池化，得到每个尺度的特征向量
                pooled = F.adaptive_avg_pool1d(conv_out, 1)  # (B, C_per_scale, 1)
                patch_features.append(pooled.squeeze(-1))  # (B, C_per_scale)
            
            # 合并当前patch的所有特征
            patch_feature = torch.cat(patch_features, dim=1)  # (B, total_channels)
            all_features.append(patch_feature)
        
        # 拼接所有patch的特征
        # 形状: (B, num_patches, total_channels)
        features = torch.stack(all_features, dim=1)
        
        # 维度调整以便批归一化
        features = features.permute(0, 2, 1)  # (B, total_channels, num_patches)
        features = self.norm(features)
        features = features.permute(0, 2, 1)  # (B, num_patches, total_channels)
        
        return self.activation(features)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
        
    def forward(self, x):
        # x形状: (B, num_patches, d_model)
        return x + self.position_embed
    
class DualAttentionTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, expansion_ratio=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 1. 局部特征增强（patch内重要信息）
        self.local_feature = nn.MultiheadAttention(
            expansion_ratio, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(expansion_ratio)

        # 2. 全局序列建模（patch间关系）
        self.global_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # 3. 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # 4. 特征变换层
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
        # 5. 聚合层
        self.aggregation = nn.Sequential(
            nn.Linear(d_model, 1),  # 降维到每个patch一个标量
            # nn.LayerNorm(num_heads)  # 保持patch数量
        )
    
    def forward(self, x):

        # 保存原始输入
        residual = x
        x = x.permute(0,2,1)
        # 1. 局部特征增强（关注patch内重要信息）
        local_out,_ = self.local_feature(x,x,x)  # (B, num_patches, d_model)
        local_out = self.norm2(local_out + x)  # 残差连接

        local_out = local_out.permute(0,2,1)
        x = x.permute(0,2,1)
        # 2. 全局序列建模（关注patch间关系）
        global_out, _ = self.global_attention(x, x, x)  # (B, num_patches, d_model)
        global_out = self.norm1(global_out + x)  # 残差连接
        
        # 3. 门控融合
        combined = torch.cat([local_out, global_out], dim=-1)
        gate = self.fusion_gate(combined)  # (B, num_patches, d_model)
        fused = gate * local_out + (1 - gate) * global_out  # (B, num_patches, d_model)
        
        # 4. 特征变换
        transformed = self.transform(fused) + residual  # 残差连接
        # print(transformed.shape)
        # 5. 聚合到每个patch一个标量
        aggregated = self.aggregation(transformed)  # (B, num_patches, 1)
        # print(aggregated.shape)

        return aggregated
    

class TShape_model(nn.Module):
    def __init__(self, seq_len, patch_size=4, num_heads=4, num_layers=3, out_channels=64):
        super().__init__()
        # 计算分块数量
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.out_channels = out_channels
        d_model = out_channels
        # 1. 分块多尺度卷积
        self.patch_conv = MultiScalePatchConvolution(
            patch_size=patch_size,
            kernel_sizes=[3],
            out_channels=out_channels
        )
        
        # 2. 位置编码
        self.pos_encoding = LearnablePositionalEncoding(
            num_patches=self.num_patches,
            d_model=out_channels
        )
        
       # Transformer层（双注意力）
        self.transformer_layers = nn.ModuleList([
            DualAttentionTransformerEncoder(
                d_model=out_channels,
                num_heads=num_heads
            )
            for _ in range(1)
        ])

        # 序列聚合层（从patch序列到单一预测）
        self.sequence_head = nn.Sequential(
            nn.Linear(4, 1),  # 保持维度兼容
        )

        # 4. 预测头
        self.head = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.GELU(),
            nn.Linear(out_channels//2, 1)
        )
        
        # 块重要性评估（可解释性）
        self.importance = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 分块与多尺度卷积
        features = self.patch_conv(x)  # (B, num_patches, out_channels)
        
        # 2. 位置编码
        features = self.pos_encoding(features)  # (B, num_patches, out_channels)
        # 应用Transformer
        for layer in self.transformer_layers:
            features = layer(features)  # (B, num_patches, 1)
    
        features = features.squeeze(2)
        prediction = self.sequence_head(features)  # (B, 1)
        
        return prediction