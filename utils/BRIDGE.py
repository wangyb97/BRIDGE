from utils.resnet import *
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.conv_layer import Conv1d, SimpleConvKAN_1layer
from torch_geometric.nn import GCNConv
from typing import List


class ADPNetblock(nn.Module):
    """
    Adaptive Pyramidal Network (ADPNet) block.

    This block performs:
        1. Constant padding for pooling.
        2. Max pooling (downsampling).
        3. Two convolutional layers with padding.
        4. Residual connection from pooled features to the output.

    Args:
        filter_num (int): Number of input/output channels (kept constant inside the block).
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor for the convolution.

    Note:
        - Padding sizes are computed to preserve spatial length after convolution.
        - Max pooling reduces the sequence length before convolution.
    """
    def __init__(self, filter_num: int, kernel_size: int, dilation: int) -> None:
        super(ADPNetblock, self).__init__()
        self.conv = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.max_pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        self.padding_conv = nn.ConstantPad1d(((kernel_size-1)//2)*dilation, 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, L_out),
                          where L_out depends on pooling/stride.
        """
        x = self.padding_pool(x)
        px = self.max_pooling(x)
        x = self.padding_conv(px)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        x = x + px
        return x


class ADPNet(nn.Module):
    """
    ADPNet for classification.

    Args:
        filter_num (int): Number of convolutional filters (channels).
        number_of_layers (int): Number of pyramid layers.
    """
    
    def __init__(self, filter_num: int, number_of_layers: int) -> None:
        super(ADPNet, self).__init__()
        
        # Predefined kernel size and dilation lists
        self.kernel_size_list: List[int] = [1 + x * 2 for x in range(number_of_layers)]
        self.kernel_size_list = [5, 5, 5, 5, 5, 5]
        self.dilation_list: List[int] = [1, 1, 1, 1, 1, 1]
        
        # Initial convolution layers
        self.conv = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )
        self.conv1 = Conv1d(
            filter_num, filter_num, self.kernel_size_list[0],
            stride=1, dilation=1, same_padding=False
        )
        # Max pooling layer for downsampling
        self.pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        
        # Constant padding layers for convolutions and pooling
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0] - 1) // 2), 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)
        
        # Pyramid blocks
        self.ADPNetblocklist = nn.ModuleList([
            ADPNetblock(
                filter_num, 
                kernel_size=self.kernel_size_list[i],
                dilation=self.dilation_list[i]
            )
            for i in range(len(self.kernel_size_list))
        ])
        
        # Final classification layer
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ADPNet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        # Initial convolution stage
        x = self.padding_conv(x)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        
        # Pyramid convolution blocks with downsampling until length <= 2
        i = 0
        while x.size()[-1] > 2:
            x = self.ADPNetblocklist[i](x)
            i += 1
            
        # Global pooling and classification
        x = x.squeeze(-1).squeeze(-1)
        logits = self.classifier(x)
        return logits


class BRIDGE(nn.Module):
    """
        BRIDGE: Multimodal sequence-structure integration network.

        This architecture processes five different modalities:
            1. Tranformer embeddings  (512 channels)
            2. Structure profiles (1 channel)
            3. Motif scores (1 channel)
            4. Biochemical features (99 channels)
            5. Graph representation of Tranformer tokens via GCN

        The outputs of each modality-specific branch are concatenated
        and fed into an ADPNet backbone for final processing.

        Args:
            k (int): Kernel size for structure and biochemical feature convs.
                    Also used to compute ADPNet depth from sequence length.
    """
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        number_of_layers = int(log(101-k+1, 2))

        # ===== Modality-specific projection layers =====
        self.conv_bert = Conv1d(512, 256, kernel_size=(1,), stride=1)
        self.conv_str = Conv1d(1, 128, kernel_size=(k,), stride=1, same_padding=True)
        self.conv_motif = Conv1d(1, 64, kernel_size=(1,), stride=1)
        self.conv_biochem = Conv1d(99, 32, kernel_size=(k,), stride=1, same_padding=True)
        
        # ===== Multiscale KAN feature extractors =====
        self.multiscale_bert = multiscaleKAN(256, 128)
        self.multiscale_str = multiscaleKAN(128, 64)
        self.multiscale_motif = multiscaleKAN(64, 32)
        self.multiscale_biochem = multiscaleKAN(32, 16)
        
        # ===== ADPNet backbone =====
        self.adpnet = ADPNet(512, number_of_layers)
        
        # ===== Graph Convolution =====
        self.gcn = GCNConv(512, 32)
        
        # ===== Initialize weights =====
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for conv, BN, and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(
        self,
        bert_embedding: torch.Tensor,  # shape: (B, 512, L)
        attn: torch.Tensor,            # shape: (B, L, L), adjacency from attention
        structure: torch.Tensor,       # shape: (B, 1, L)
        motif: torch.Tensor,           # shape: (B, 1, M)
        biochem: torch.Tensor           # shape: (B, 99, L)
    ) -> torch.Tensor:
        """
        Forward pass of BRIDGE.

        Args:
            bert_embedding: BERT-derived embeddings for each token.
            attn: Attention weight matrices (used as adjacency).
            structure: Structure probability profiles.
            motif: Motif prior scores (may be shorter than sequence length).
            biochem: Biochemical feature profiles.

        Returns:
            torch.Tensor: Output of ADPNet after multimodal integration.
        """
        
        node_features = bert_embedding
        adj = attn
        batch_size, num_nodes, _ = adj.shape
        edge_index_list = []
        for i in range(batch_size):
            edge_index = adj[i].nonzero(as_tuple=False).t().contiguous()
            edge_index_list.append(edge_index)
        edge_index = torch.cat(edge_index_list, dim=1)
        node_features = node_features.permute(0, 2, 1).contiguous().view(-1, 512)
        x = self.gcn(node_features, edge_index)
        x = x.view(batch_size, num_nodes, -1).permute(0, 2, 1).contiguous()
        
        x0 = self.conv_bert(bert_embedding)
        x0 = self.multiscale_bert(x0)
        
        x1 = structure
        x1 = self.conv_str(x1)
        x1 = self.multiscale_str(x1)
        
        x2 = motif
        x2 = self.conv_motif(x2)
        x2 = self.multiscale_motif(x2)
        total_padding = 101 - x2.size(2)
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad
        x2 = F.pad(x2, (left_pad, right_pad), "constant", 0)

        x3 = self.conv_biochem(biochem)
        x3 = self.multiscale_biochem(x3)

        x = torch.cat([x, x0, x1, x2, x3], dim=1)
        return self.adpnet(x)
    

class multiscaleKAN(nn.Module):
    """
    Multi-scale KAN block with residual connection.

    This module applies:
        - Path 0: A single 1x1 SimpleConvKAN layer.
        - Path 1: A 1x1 SimpleConvKAN followed by a 3x3 SimpleConvKAN.

    The outputs of both paths are concatenated along the channel dimension
    and then added to the original input (residual connection).

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels per path.

    Note:
        Final output channel dimension = in_channel + 2 * out_channel
        only if the residual input's channel size matches the concatenated output's channel size.
    """
    
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(multiscaleKAN, self).__init__()

        self.conv0 = SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, grid_size=2, dropout=0.3)
        self.conv1 = nn.Sequential(
            SimpleConvKAN_1layer(in_channel, out_channel, kernel_size=1, same_padding=False, bn=False, grid_size=2, dropout=0.3),
            SimpleConvKAN_1layer(out_channel, out_channel, kernel_size=3, same_padding=True, grid_size=4, dropout=0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L)

        Returns:
            torch.Tensor: Output tensor after multi-scale feature extraction
                          and residual addition. Shape is (B, C_out, L),
                          where C_out = C + 2*out_channel if concatenation
                          changes channels, else same as input.
        """
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        return torch.cat([x0,x1], dim=1) + x
