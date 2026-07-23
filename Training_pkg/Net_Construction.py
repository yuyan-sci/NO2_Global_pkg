import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from Training_pkg.utils import *
import torch.nn.functional as F

import lightgbm as lgb

activation = activation_function_table()

def resnet_block_lookup_table(blocktype):
    if blocktype == 'BasicBlock':
        return BasicBlock
    elif blocktype == 'Bottleneck':
        return Bottleneck
    else:
        print(' Wrong Key Word! BasicBlock or Bottleneck only! ')
        return None

def initial_network(width,main_stream_nchannel,side_stream_nchannel, blocks_num=None):

    if TwoCombineModels_Settings:
        Model_A = initial_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel)
        Model_B = initial_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel)
        cnn_model = Combine_GeophysicalDivide_Two_Models(model_A=Model_A, model_B=Model_B)
    else:
        cnn_model = initial_OneStage_network(width=width,main_stream_nchannel=main_stream_nchannel,side_stream_nchannel=side_stream_nchannel, blocks_num=blocks_num)
    return cnn_model

def initial_OneStage_network(width,main_stream_nchannel,side_stream_nchannel, blocks_num=None):
    if ResNet_setting:
        block = resnet_block_lookup_table(ResNet_Blocks)
        if blocks_num is None:
            cnn_model = ResNet(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
        else:
            cnn_model = ResNet(nchannel=main_stream_nchannel,block=block,blocks_num=blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)    
    elif ResNet_MLP_setting:
        block = resnet_block_lookup_table(ResNet_MLP_Blocks)
        cnn_model = ResNet_MLP(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_MLP_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)#cnn_model = Net(nchannel=nchannel)
    elif ResNet_Classification_Settings:
        block = resnet_block_lookup_table(ResNet_Classification_Blocks)
        cnn_model = ResNet_Classfication(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_Classification_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
    elif ResNet_MultiHeadNet_Settings:
        block = resnet_block_lookup_table(ResNet_MultiHeadNet_Blocks)
        cnn_model = MultiHead_ResNet(nchannel=main_stream_nchannel,block=block,blocks_num=ResNet_MultiHeadNet_blocks_num,include_top=True,groups=1,width_per_group=width)
    elif LateFusion_setting:
        block = resnet_block_lookup_table(LateFusion_Blocks)
        cnn_model = LateFusion_ResNet(nchannel=main_stream_nchannel,nchannel_lf=side_stream_nchannel,block=block,blocks_num=LateFusion_blocks_num,num_classes=1,include_top=True,groups=1,width_per_group=width)
    elif MultiHeadLateFusion_settings:
        block = resnet_block_lookup_table(MultiHeadLateFusion_Blocks)
        cnn_model = MultiHead_LateFusion_ResNet(nchannel=main_stream_nchannel,nchannel_lf=side_stream_nchannel,block=block,blocks_num=MultiHeadLateFusion_blocks_num,include_top=True,groups=1,width_per_group=width)
    elif UNet_setting:
        # Note: 'width' parameter is spatial width (5), not network capacity
        # Use 64 as default network capacity (matches ResNet baseline)
        network_width = 64
        cnn_model = UNet_GlobalPool(nchannel=main_stream_nchannel,num_classes=1,width_per_group=network_width)
    return cnn_model
        
class BasicBlock(nn.Module):  
    
    expansion = 1  
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, activation='tanh', **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if activation == 'relu':
            self.actfunc = nn.ReLU()
        elif activation == 'tanh':
            self.actfunc = nn.Tanh()
        elif activation == 'gelu':
            self.actfunc = nn.GELU()
        elif activation == 'sigmoid':
            self.actfunc = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.downsample = downsample
        
    def forward(self, x):
        
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity # out=F(X)+X
        out = self.actfunc(out)

        return out
    
class Bottleneck(nn.Module):  
    # Three convolutional layers, F(x) and X have different dimensions.
    """
    注意: 原论文中, 在虚线残差结构的主分支上, 第一个1x1卷积层的步距是2, 第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1, 第二个3x3卷积层步距是2,
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, activation='tanh', width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        global ReLU_ACF, Tanh_ACF, GeLU_ACF, Sigmoid_ACF
        if activation == 'relu':
            self.actfunc = nn.ReLU()
        elif activation == 'tanh':
            self.actfunc = nn.Tanh()
        elif activation == 'gelu':
            self.actfunc = nn.GELU()
        elif activation == 'sigmoid':
            self.actfunc = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.downsample = downsample

    def forward(self, x):
        in_size = x.size(0)
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.actfunc(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.actfunc(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out=F(X)+X
        out = out + identity
        out = self.actfunc(out)

        return out

class DropBlock2D(nn.Module):
    """DropBlock implementation for 2D feature maps"""
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        
        # Get dimensions
        N, C, H, W = x.size()
        
        # Calculate gamma (number of blocks to drop)
        gamma = self.drop_rate / (self.block_size ** 2)
        
        # Sample mask
        mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
        
        # Pad mask to match input size
        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
        
        # Create block mask
        block_mask = 1 - F.max_pool2d(
            mask, 
            kernel_size=self.block_size, 
            stride=1, 
            padding=self.block_size // 2
        )
        
        # Normalize to maintain expected values
        normalize_factor = block_mask.numel() / block_mask.sum()
        
        return x * block_mask * normalize_factor
   
class ResNet(nn.Module):

    def __init__(self,
                nchannel, # initial input channel
                block,  # block types
                blocks_num,  
                num_classes=1,  
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  
        self.dropout_rate = dropout_rate
        self.fc_dropout_rate = fc_dropout_rate
        self.dropblock_rate = dropblock_rate

        self.groups = groups
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.width_per_group = width_per_group
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6

        if standard_dropout:
            self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False)
                                        ,nn.BatchNorm2d(self.in_channel)
                                        ,self.actfunc
                                        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                        ,nn.Dropout2d(p=dropout_rate)) # output 4x4
            if He_initialization:
                nn.init.kaiming_normal_(self.layer0[0].weight, mode='fan_out', nonlinearity='tanh')

            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.dropout1 = nn.Dropout2d(p=dropout_rate) 
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
            self.dropout2 = nn.Dropout2d(p=dropout_rate)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
            self.dropout3 = nn.Dropout2d(p=dropout_rate)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)
            self.dropout4 = nn.Dropout2d(p=dropout_rate)
        
        elif DropBlock:
            self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False)
                                        ,nn.BatchNorm2d(self.in_channel)
                                        ,self.actfunc
                                        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1))     
            if He_initialization:
                nn.init.kaiming_normal_(self.layer0[0].weight, mode='fan_out', nonlinearity='tanh')

            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

            self.dropblock1 = DropBlock2D(drop_rate=dropblock_rate, block_size=dropblock_size)
            self.dropblock2 = DropBlock2D(drop_rate=dropblock_rate, block_size=dropblock_size)
            self.dropblock3 = DropBlock2D(drop_rate=dropblock_rate, block_size=dropblock_size)
            self.dropblock4 = DropBlock2D(drop_rate=dropblock_rate, block_size=dropblock_size)
        
        else:
            self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False)
                                        ,nn.BatchNorm2d(self.in_channel)
                                        ,self.actfunc
                                        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1))      
            if He_initialization:
                nn.init.kaiming_normal_(self.layer0[0].weight, mode='fan_out', nonlinearity='tanh')

            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)  # Dropout before FC layer
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        # Apply regularization after layer1
        if standard_dropout:
            print("Using standard dropout")
            x = self.dropout1(x)
        elif DropBlock:
            print("Using DropBlock")
            print(hasattr(self, 'dropblock1'))
            x = self.dropblock1(x)
        
        x = self.layer2(x)
        if standard_dropout:
            x = self.dropout2(x)
        elif DropBlock:
            x = self.dropblock2(x)
        
        x = self.layer3(x)
        if standard_dropout:
            x = self.dropout3(x)
        elif DropBlock:
            x = self.dropblock3(x)
        
        x = self.layer4(x)
        if standard_dropout:
            x = self.dropout4(x)
        elif DropBlock:
            x = self.dropblock4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if standard_dropout:
                x = self.fc_dropout(x)
            x = self.fc(x)
            
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # if self.include_top:  
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     #x = self.actfunc(x)
        #     x = self.fc(x)

        return x

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure reduced channels is at least 1
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class UNetConvBlock(nn.Module):
    """Configurable UNet convolutional block (plain or residual)"""
    def __init__(self, in_channels, out_channels, num_blocks=2, use_residual=False, activation='gelu'):
        super(UNetConvBlock, self).__init__()
        self.use_residual = use_residual
        
        # Get activation function
        if activation == 'relu':
            actfunc = nn.ReLU()
        elif activation == 'tanh':
            actfunc = nn.Tanh()
        elif activation == 'gelu':
            actfunc = nn.GELU()
        elif activation == 'sigmoid':
            actfunc = nn.Sigmoid()
        else:
            actfunc = nn.GELU()
        
        layers = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                actfunc
            ])
        self.conv_block = nn.Sequential(*layers)
        
        # Residual connection if needed
        if use_residual and in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif use_residual:
            self.residual = nn.Identity()
        else:
            self.residual = None
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out

class UNet_GlobalPool(nn.Module):
    """
    Configurable UNet-style Encoder-Decoder with Global Pooling.
    Supports multiple architecture configurations through config.toml.
    """
    def __init__(self,
                 nchannel,
                 num_classes=1,
                 width_per_group=64,
                 include_top=True):
        
        super(UNet_GlobalPool, self).__init__()
        self.include_top = include_top
        
        # Get configuration from global settings
        self.depth = UNet_depth
        self.use_residual = UNet_use_residual_blocks
        self.blocks_per_level = UNet_blocks_per_level
        self.use_attention = UNet_use_attention
        self.bottleneck_type = UNet_bottleneck_type
        self.remove_pooling = UNet_remove_pooling
        self.decoder_channels = UNet_decoder_channels
        
        # Get activation function name
        if ReLU_ACF:
            self.act_name = 'relu'
        elif Tanh_ACF:
            self.act_name = 'tanh'
        elif GeLU_ACF:
            self.act_name = 'gelu'
        elif Sigmoid_ACF:
            self.act_name = 'sigmoid'
        else:
            self.act_name = 'gelu'
        
        # Base features
        base_features = int(64 * (width_per_group / 64.))
        
        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        # Initial conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(nchannel, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            self._get_activation()
        )
        
        # Encoder layers
        in_ch = base_features
        self.encoder_channels = [base_features]
        for i in range(self.depth):
            out_ch = base_features * (2 ** i)
            self.encoder_channels.append(out_ch)
            
            # Encoder block
            self.encoder_blocks.append(
                UNetConvBlock(in_ch, out_ch, self.blocks_per_level, self.use_residual, self.act_name)
            )
            
            # Downsampling
            if i < self.depth:  # Don't downsample after last encoder block
                if self.remove_pooling:
                    self.downsample_blocks.append(
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
                    )
                else:
                    self.downsample_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_features * (2 ** self.depth)
        if self.bottleneck_type == 'mlp':
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.encoder_channels[-1], bottleneck_ch),
                self._get_activation(),
                nn.Linear(bottleneck_ch, bottleneck_ch),
                self._get_activation()
            )
            self.bottleneck_reshape = True
        else:
            self.bottleneck = UNetConvBlock(
                self.encoder_channels[-1], bottleneck_ch, 
                self.blocks_per_level, self.use_residual, self.act_name
            )
            self.bottleneck_reshape = False
        
        # Attention at bottleneck
        if self.use_attention and not self.bottleneck_reshape:
            self.bottleneck_attention = nn.Sequential(
                ChannelAttention(bottleneck_ch),
                SpatialAttention()
            )
        else:
            self.bottleneck_attention = None
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        # Track decoder output channels for building subsequent layers
        prev_dec_out_ch = bottleneck_ch
        
        # Build skip attentions first (in forward order to match skip_connections)
        if self.use_attention:
            self.skip_attentions = nn.ModuleList()
            # Skip connections are stored as [enc_block_0_output, enc_block_1_output, ...]
            # which have channels [encoder_channels[1], encoder_channels[2], ...]
            for i in range(self.depth):
                self.skip_attentions.append(ChannelAttention(self.encoder_channels[i+1]))
        else:
            self.skip_attentions = None
        
        for i in range(self.depth - 1, -1, -1):
            # Decoder channels
            if self.decoder_channels == 'half':
                dec_out_ch = self.encoder_channels[i] // 2
            else:
                dec_out_ch = self.encoder_channels[i]
            
            # Upsample layer (reduces channels to match skip connection level)
            upsample_out_ch = self.encoder_channels[i+1]
            self.upsample_blocks.append(
                nn.ConvTranspose2d(prev_dec_out_ch, upsample_out_ch, kernel_size=2, stride=2, bias=False)
            )
            
            # Input channels = upsampled + skip connection
            # Skip connections have encoder_channels[i+1] channels
            dec_in_ch = upsample_out_ch + self.encoder_channels[i+1]
            
            # Decoder block
            self.decoder_blocks.append(
                UNetConvBlock(dec_in_ch, dec_out_ch, self.blocks_per_level, self.use_residual, self.act_name)
            )
            
            # Update prev_dec_out_ch for next iteration
            prev_dec_out_ch = dec_out_ch
        
        # Regularization
        if standard_dropout:
            self.dropout_final = nn.Dropout2d(p=dropout_rate)
        elif DropBlock:
            self.dropblock_final = DropBlock2D(drop_rate=dropblock_rate, block_size=dropblock_size)
        else:
            self.dropout_final = None
        
        # Output head
        if self.include_top:
            # Use the last decoder output channels (stored in prev_dec_out_ch)
            final_ch = prev_dec_out_ch
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)
            self.fc = nn.Linear(final_ch, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"\n{'='*70}")
        print(f"UNet Configuration:")
        print(f"  Depth: {self.depth}")
        print(f"  Blocks per level: {self.blocks_per_level}")
        print(f"  Residual blocks: {self.use_residual}")
        print(f"  Attention: {self.use_attention}")
        print(f"  Bottleneck: {self.bottleneck_type}")
        print(f"  Pooling: {'Strided Conv' if self.remove_pooling else 'MaxPool'}")
        print(f"  Decoder channels: {self.decoder_channels}")
        print(f"  Encoder channels: {self.encoder_channels}")
        print(f"{'='*70}\n")
    
    def _get_activation(self):
        if ReLU_ACF:
            return nn.ReLU()
        elif Tanh_ACF:
            return nn.Tanh()
        elif GeLU_ACF:
            return nn.GELU()
        elif Sigmoid_ACF:
            return nn.Sigmoid()
        return nn.GELU()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            if i < self.depth - 1:
                x = self.downsample_blocks[i](x)
        
        # Bottleneck
        if self.bottleneck_reshape:
            # MLP bottleneck
            b, c, h, w = x.size()
            x = self.bottleneck(x)
            # Reshape back to spatial
            spatial_size = skip_connections[-1].size()[2:]
            x = x.view(b, -1, 1, 1).expand(-1, -1, spatial_size[0], spatial_size[1])
        else:
            x = self.bottleneck(x)
            if self.bottleneck_attention is not None:
                x = self.bottleneck_attention(x)
        
        # Decoder with skip connections
        for i in range(self.depth):
            # Upsample
            x = self.upsample_blocks[i](x)
            
            # Get skip connection (in reverse order from encoder)
            skip = skip_connections[-(i+1)]
            
            # Apply attention to skip if enabled
            # Note: skip_attentions are in forward order, but we access skips in reverse
            if self.use_attention and self.skip_attentions is not None:
                skip_attention_idx = self.depth - 1 - i
                skip = self.skip_attentions[skip_attention_idx](skip)
            
            # Match spatial dimensions
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Decoder block
            x = self.decoder_blocks[i](x)
        
        # Regularization
        if self.dropout_final is not None:
            x = self.dropout_final(x)
        
        # Output head
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc_dropout(x)
            x = self.fc(x)
        
        return x

class Combine_GeophysicalDivide_Two_Models(nn.Module):
    def __init__(self,model_A,model_B,):
        super(Combine_GeophysicalDivide_Two_Models, self).__init__()
        self.model_A = model_A
        self.model_B = model_B
    def forward(self,x_A,x_B):
        x_A = self.model_A(x_A)
        x_B = self.model_B(x_B)
        return x_A, x_B

class ResNet_MLP(nn.Module):
    
    def __init__(self,
                nchannel, # initial input channel
                block,  # block types
                blocks_num,  
                num_classes=1,  
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(ResNet_MLP, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        # self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=3, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

        self.mlp_outlayer = nn.Sequential(nn.Linear(512 * block.expansion,512),
                                        self.actfunc,
                                        nn.BatchNorm1d(512),
                                        nn.Linear(512,128),
                                        self.actfunc,
                                        nn.Linear(128,num_classes))

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            #x = self.fc(x)
            x = self.mlp_outlayer(x)

        return x
    
class ResNet_Classfication(nn.Module):

    def __init__(self,
                nchannel, # initial input channel
                block,  # block types
                blocks_num,  
                num_classes=1,  
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(ResNet_Classfication, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = ResNet_Classification_left_bin
        self.right_bin   = ResNet_Classification_right_bin
        self.bins_number = ResNet_Classification_bins_number
        #self.conv1 = nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.in_channel)

        #self.tanh = nn.Tanh()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer0 = nn.Sequential(self.conv1,self.bn1,self.tanh,self.maxpool)
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        self.softmax = nn.Softmax()
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        return classification_output

class MultiHead_ResNet(nn.Module):
    
    def __init__(self,
                nchannel, # initial input channel
                block,  # block types
                blocks_num,   
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(MultiHead_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_cls = 64  


        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = ResNet_MultiHeadNet_left_bin
        self.right_bin   = ResNet_MultiHeadNet_right_bin
        self.bins_number = ResNet_MultiHeadNet_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=1)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1)


        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            regression_output = self.fc(x)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        return regression_output, classification_output
    
class LateFusion_ResNet(nn.Module):

    def __init__(self,
                nchannel, # initial input channel
                nchannel_lf, # input channel for late fusion
                block,  # block types
                blocks_num,  
                num_classes=1,  
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(LateFusion_ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        
        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        #self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=5, stride=1,padding=1, bias=False)
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)
        
                
        
        

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
            
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf+self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_lf = self.layer0_lf(x_lf)
        x_lf = self.layer1_lf(x_lf)
        x_lf = self.layer2_lf(x_lf)
        x_lf = self.layer3_lf(x_lf)
        
        
        x = torch.cat((x,x_lf),1)
        x = self.fuse_layer(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            x = self.fc(x)

        return x
     
'''
class MultiHead_LateFusion_ResNet(nn.Module):
    
    def __init__(self,
                nchannel, # initial input channel
                nchannel_lf, # input channel for late fusion
                block,  # block types
                blocks_num,   
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(MultiHead_LateFusion_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.groups = groups
        self.width_per_group = width_per_group
        self.actfunc = activation_func
        self.left_bin    = MultiHeadLateFusion_left_bin
        self.right_bin   = MultiHeadLateFusion_right_bin
        self.bins_number = MultiHeadLateFusion_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,activation_func
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)
        

        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel_lf = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel_lf,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        
        layers.append(block(self.in_channel_lf+self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion # The input channel changed here!``
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_lf = self.layer0_lf(x_lf)
        x_lf = self.layer1_lf(x_lf)
        x_lf = self.layer2_lf(x_lf)
        x_lf = self.layer3_lf(x_lf)
        
        
        x = torch.cat((x,x_lf),1)
        x = self.fuse_layer(x)

        if self.include_top:  
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            #x = self.actfunc(x)
            regression_output = self.fc(x)
            classification_output = self.bins_fc(x)
            classification_output = self.softmax(classification_output)
        #final_output = 0.5*regression_output + 0.5*torch.matmul(classification_output,self.bins)
        return regression_output, classification_output
'''      
class MultiHead_LateFusion_ResNet(nn.Module):
    
    def __init__(self,
                nchannel, # initial input channel
                nchannel_lf, # input channel for late fusion
                block,  # block types
                blocks_num,   
                include_top=True, 
                groups=1,
                width_per_group=64):

        super(MultiHead_LateFusion_ResNet, self).__init__()
        
        self.include_top = include_top
        self.in_channel = 64  
        self.in_channel_lf = 16
        self.in_channel_clsfy = 64  
        self.in_channel_lf_clsfy = 16  

        self.groups = groups
        self.width_per_group = width_per_group
        if ReLU_ACF == True:
            self.actfunc =  nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        self.left_bin    = MultiHeadLateFusion_left_bin
        self.right_bin   = MultiHeadLateFusion_right_bin
        self.bins_number = MultiHeadLateFusion_bins_number
        self.bins        = torch.tensor(np.linspace(self.left_bin,self.right_bin,self.bins_number))

        self.layer0 = nn.Sequential(nn.Conv2d(nchannel, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=1)
        self.layer3 = self._make_layer(block, 256, blocks_num[3], stride=1)

        self.layer1_lf = self._make_layer_lf(block, 32, blocks_num[0])
        self.layer2_lf = self._make_layer_lf(block, 64, blocks_num[1], stride=1)
        self.layer3_lf = self._make_layer_lf(block, 64, blocks_num[2], stride=1)

        self.fuse_layer = self._make_layer_fused(block, 512, blocks_num[2], stride=1)


        self.layer0_clsfy = nn.Sequential(nn.Conv2d(nchannel, self.in_channel_clsfy, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_clsfy)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4

        self.layer0_lf_clsfy = nn.Sequential(nn.Conv2d(nchannel_lf, self.in_channel_lf_clsfy, kernel_size=7, stride=2,padding=3, bias=False) #output size:6x6
        ,nn.BatchNorm2d(self.in_channel_lf_clsfy)
        ,self.actfunc
        ,nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # output 4x4
        
        self.layer1_clsfy = self._make_layer_clsfy(block, 64, blocks_num[0])
        self.layer2_clsfy = self._make_layer_clsfy(block, 128, blocks_num[1], stride=1)
        self.layer3_clsfy = self._make_layer_clsfy(block, 256, blocks_num[3], stride=1)

        self.layer1_lf_clsfy = self._make_layer_lf_clsfy(block, 32, blocks_num[0])
        self.layer2_lf_clsfy = self._make_layer_lf_clsfy(block, 64, blocks_num[1], stride=1)
        self.layer3_lf_clsfy = self._make_layer_lf_clsfy(block, 64, blocks_num[2], stride=1)

        self.fuse_layer_clsfy = self._make_layer_fused_clsfy(block, 512, blocks_num[2], stride=1)
        


        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            self.fc = nn.Linear(512 * block.expansion, 1)
            self.bins_fc = nn.Linear(512 * block.expansion, self.bins_number)
        
        self.softmax = nn.Softmax()

        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation_func_name)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf+self.in_channel) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf+self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf+self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    def _make_layer_clsfy(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_clsfy != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def _make_layer_lf_clsfy(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel_lf_clsfy != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_lf_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_lf_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def _make_layer_fused_clsfy(self, block, channel, block_num, stride=1):
        if stride != 1 or (self.in_channel_lf_clsfy+self.in_channel_clsfy) != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel_lf_clsfy+self.in_channel_clsfy, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        if block_num == 0:
            layers.append(nn.Identity())
        else:
            layers.append(block(self.in_channel_lf_clsfy+self.in_channel_clsfy,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                activation=activation,
                                width_per_group=self.width_per_group))

            self.in_channel_clsfy = channel * block.expansion # The input channel changed here!``
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel_clsfy,
                                    channel,
                                    groups=self.groups,
                                    activation=activation,
                                    width_per_group=self.width_per_group))
        return nn.Sequential(*layers)
    
    def forward(self, x,x_lf):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.tanh(x)
        #x = self.maxpool(x)

        x_r = self.layer0(x)
        x_r = self.layer1(x_r)
        x_r = self.layer2(x_r)
        x_r = self.layer3(x_r)

        x_lf_r = self.layer0_lf(x_lf)
        x_lf_r = self.layer1_lf(x_lf_r)
        x_lf_r = self.layer2_lf(x_lf_r)
        x_lf_r = self.layer3_lf(x_lf_r)
        
        
        x_r = torch.cat((x_r,x_lf_r),1)
        x_r = self.fuse_layer(x_r)

        #######################################
        x_c = self.layer0_clsfy(x)
        x_c = self.layer1_clsfy(x_c)
        x_c = self.layer2_clsfy(x_c)
        x_c = self.layer3_clsfy(x_c)

        x_lf_c = self.layer0_lf_clsfy(x_lf)
        x_lf_c = self.layer1_lf_clsfy(x_lf_c)
        x_lf_c = self.layer2_lf_clsfy(x_lf_c)
        x_lf_c = self.layer3_lf_clsfy(x_lf_c)

        x_c = torch.cat((x_c,x_lf_c),1)
        x_c = self.fuse_layer_clsfy(x_c)

        if self.include_top:  
            x_r = self.avgpool(x_r)
            x_r = torch.flatten(x_r, 1)
            x_c = self.avgpool(x_c)
            x_c = torch.flatten(x_c, 1)
            #x = self.actfunc(x)
            regression_output = self.fc(x_r)
            classification_output = self.bins_fc(x_c)
            classification_output = self.softmax(classification_output)
        #final_output = 0.5*regression_output + 0.5*torch.matmul(classification_output,self.bins)
        return regression_output, classification_output
    
class Net(nn.Module):
    def __init__(self, nchannel):
        super(Net, self).__init__()

        self.conv = nn.Sequential(  # The first loop of ConvLay er, ReLU, Pooling
            nn.Conv2d(in_channels=nchannel,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=0),
            nn.BatchNorm2d(64,momentum=0.1),
            nn.Tanh(),

            # self.conv2 = nn.Sequential(ResidualBlocks(in_channel=32,out_channel=32,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=0),
            nn.BatchNorm2d(128,momentum=0.1),
            nn.Tanh(),

            # self.conv4 = nn.Sequential(ResidualBlocks(in_channel=64,out_channel=64,kernel_size=3,stride=1,padding=1),
            #                           nn.Tanh())

            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=0),
            nn.BatchNorm2d(256,momentum=0.1),
            nn.Tanh(),
            # The first loop of ConvLayer, ReLU, Pooling
            nn.Conv2d(in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.BatchNorm2d(512,momentum=0.1),
            nn.Tanh()
        )

        # self.ful1 = nn.Sequential(nn.Linear(256  * 5 * 5, 64), nn.BatchNorm1d(64))
        self.ful = nn.Sequential(nn.Linear(512 * 5 * 5, 64),  # , nn.BatchNorm1d(64), nn.Tanh())
                                nn.Linear(64, 16),  # ,nn.BatchNorm1d(16), nn.Tanh())  # ,nn.Softmax())
                                nn.Linear(16, 2),  # ,nn.BatchNorm1d(2), nn.Tanh())
                                nn.Linear(2, 1))

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv(x)
        out = out.view(in_size, -1)
        output = self.ful(out)
        return output

class PhysicsInformedNO2Net(nn.Module):
    def __init__(self, 
                 feature_names,  # List of feature names
                 num_classes=1,
                 include_top=True):
        super(PhysicsInformedNO2Net, self).__init__()
        
        self.include_top = include_top
        self.feature_names = feature_names
        
        # Create feature name to index mapping
        self.feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Define feature groups based on atmospheric physics
        self.emission_features = ['Population', 'NO_05_emi', 'Total_DM']
        self.transport_features = ['V10M', 'U10M', 'USTAR', 'PBLH', 'PS']
        self.chemistry_features = ['T2M', 'RH', 'TSW', 'TP']
        self.surface_features = ['NDVI', 'ISA', 'forests_density', 'shrublands_distance', 
                                'croplands_distance', 'urban_builtup_lands_buffer-6500', 'water_bodies_distance']
        self.geographic_features = ['elevation', 'x', 'y', 'z', 'log_major_roads', 'log_minor_roads_new']
        self.satellite_features = ['GeoNO2', 'GCHP_NO2']
        
        # Activation function (matching your existing code style)
        if ReLU_ACF == True:
            self.actfunc = nn.ReLU()
        elif Tanh_ACF == True:
            self.actfunc = nn.Tanh()
        elif GeLU_ACF == True:
            self.actfunc = nn.GELU()
        elif Sigmoid_ACF == True:
            self.actfunc = nn.Sigmoid()
        
        # Get feature counts for each domain
        self.n_emission = len(self.emission_features)
        self.n_transport = len(self.transport_features)
        self.n_chemistry = len(self.chemistry_features)
        self.n_surface = len(self.surface_features)
        self.n_geographic = len(self.geographic_features)
        self.n_satellite = len(self.satellite_features)
        
        # Validate that all features are present
        self._validate_features()
        
        # Emission source processing
        self.emission_processor = nn.Sequential(
            nn.Linear(self.n_emission, 64),
            self.actfunc,
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            self.actfunc,
            nn.BatchNorm1d(32)
        )
        
        # Transport/dispersion processing
        self.transport_processor = nn.Sequential(
            nn.Linear(self.n_transport, 64),
            self.actfunc,
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            self.actfunc,
            nn.BatchNorm1d(32)
        )
        
        # Chemical processing
        self.chemistry_processor = nn.Sequential(
            nn.Linear(self.n_chemistry, 64),
            self.actfunc,
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            self.actfunc,
            nn.BatchNorm1d(32)
        )
        
        # Surface interaction processing
        self.surface_processor = nn.Sequential(
            nn.Linear(self.n_surface, 64),
            self.actfunc,
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            self.actfunc,
            nn.BatchNorm1d(32)
        )
        
        # Geographic/coordinate processing
        self.geographic_processor = nn.Sequential(
            nn.Linear(self.n_geographic, 32),
            self.actfunc,
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            self.actfunc
        )
        
        # Satellite/remote sensing processing
        self.satellite_processor = nn.Sequential(
            nn.Linear(self.n_satellite, 32),
            self.actfunc,
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            self.actfunc
        )
        
        # Physics-based combination with attention mechanism
        total_features = 32 + 32 + 32 + 32 + 16 + 16  # 160 total
        
        # Attention layer to weight different physical processes
        self.attention = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.Tanh(),
            nn.Linear(64, 6),  # 6 attention weights for 6 processors
            nn.Softmax(dim=1)
        )
        
        # Physics-based combination
        self.physics_combiner = nn.Sequential(
            nn.Linear(total_features, 128),
            self.actfunc,
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            self.actfunc,
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            self.actfunc,
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _validate_features(self):
        """Validate that all required features are present in the feature list"""
        all_required = (self.emission_features + self.transport_features + 
                       self.chemistry_features + self.surface_features + 
                       self.geographic_features + self.satellite_features)
        
        missing_features = [f for f in all_required if f not in self.feature_names]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        print(f"Feature counts: Emission={self.n_emission}, Transport={self.n_transport}, "
              f"Chemistry={self.n_chemistry}, Surface={self.n_surface}, "
              f"Geographic={self.n_geographic}, Satellite={self.n_satellite}")
    
    def _get_feature_indices(self, feature_group):
        """Get indices for a group of features"""
        return [self.feature_to_idx[name] for name in feature_group if name in self.feature_to_idx]
    
    def _initialize_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x should be flattened feature vector: (batch_size, n_features)
        batch_size = x.size(0)
        
        # Extract features by name using indices
        emission_indices = self._get_feature_indices(self.emission_features)
        transport_indices = self._get_feature_indices(self.transport_features)
        chemistry_indices = self._get_feature_indices(self.chemistry_features)
        surface_indices = self._get_feature_indices(self.surface_features)
        geographic_indices = self._get_feature_indices(self.geographic_features)
        satellite_indices = self._get_feature_indices(self.satellite_features)
        
        # Select features by name
        emission_features = x[:, emission_indices]
        transport_features = x[:, transport_indices]
        chemistry_features = x[:, chemistry_indices]
        surface_features = x[:, surface_indices]
        geographic_features = x[:, geographic_indices]
        satellite_features = x[:, satellite_indices]
        
        # Process each physics domain
        emission_out = self.emission_processor(emission_features)
        transport_out = self.transport_processor(transport_features)
        chemistry_out = self.chemistry_processor(chemistry_features)
        surface_out = self.surface_processor(surface_features)
        geographic_out = self.geographic_processor(geographic_features)
        satellite_out = self.satellite_processor(satellite_features)
        
        # Concatenate all processed features
        combined_features = torch.cat([
            emission_out, transport_out, chemistry_out, 
            surface_out, geographic_out, satellite_out
        ], dim=1)
        
        # Optional: Apply attention weighting
        if hasattr(self, 'attention'):
            attention_weights = self.attention(combined_features)
            # Apply attention to each domain separately
            weighted_features = torch.cat([
                emission_out * attention_weights[:, 0:1],
                transport_out * attention_weights[:, 1:2],
                chemistry_out * attention_weights[:, 2:3],
                surface_out * attention_weights[:, 3:4],
                geographic_out * attention_weights[:, 4:5],
                satellite_out * attention_weights[:, 5:6]
            ], dim=1)
            combined_features = weighted_features
        
        # Final physics-based prediction
        output = self.physics_combiner(combined_features)
        
        return output

class LightGBMModel:
    def __init__(self, params=None):
        """
        Initialize LightGBM model with parameters from config,
        aligned to study text (GOSS / EFB / histogram / leaf-wise + regularization).
        """
        if params is None:
            # ---- read globals from your existing config parses (see section 4 below) ----
            # Required (already in your code)
            objective        = LightGBM_objective
            metric           = LightGBM_metric
            boosting_type    = LightGBM_boosting_type
            data_sample_strategy = LightGBM_data_sample_strategy
            device_type      = LightGBM_device            # "cpu" or "gpu" in your cfg
            num_leaves       = int(LightGBM_num_leaves)
            learning_rate    = float(LightGBM_learning_rate)
            feature_fraction = float(LightGBM_feature_fraction)
            verbose          = int(LightGBM_verbose)

            # New optional knobs (safe defaults for study setup)
            top_rate         = float(LightGBM_top_rate)     # GOSS
            other_rate       = float(LightGBM_other_rate)   # GOSS
            max_bin          = int(LightGBM_max_bin)         # histogram
            lambda_l1        = float(LightGBM_lambda_l1)
            lambda_l2        = float(LightGBM_lambda_l2)
            min_data_in_leaf = int(LightGBM_min_data_in_leaf)
            min_gain_to_split= float(LightGBM_min_gain_to_split)
            max_depth        = int(LightGBM_max_depth)
            enable_bundle    = bool(LightGBM_enable_bundle)
            num_threads      = int(LightGBM_num_threads)       # 0 = all cores

            bagging_fraction = float(LightGBM_bagging_fraction)
            bagging_freq = int(LightGBM_bagging_freq)

            self.params = {
                "objective": objective,
                "metric": metric,
                "boosting_type": boosting_type,
                "data_sample_strategy": data_sample_strategy,
                "device_type": device_type,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "feature_fraction": feature_fraction,

                # Sampling / GOSS / histogram / EFB
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "top_rate": top_rate,
                "other_rate": other_rate,
                "max_bin": max_bin,
                "enable_bundle": enable_bundle,

                # Regularization & constraints (Ω(f_t))
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "min_data_in_leaf": min_data_in_leaf,
                "min_gain_to_split": min_gain_to_split,
                "max_depth": max_depth,

                # Runtime
                "num_threads": num_threads,
                "verbose": verbose,
                "force_col_wise": True,   # good for wide features
                "deterministic": LightGBM_deterministic,
            }
        else:
            self.params = params

        self.model = None
        self.feature_importance = None
        self.best_iteration = 0

    def flatten_features(self, X, mode="center"):
        """
        Convert (N, C, H, W) -> tabular.
        mode='center': use center pixel (fast, your current behavior)
        mode='flatten': use all H*W per channel (may get very wide)
        """
        if X.ndim == 4:
            N, C, H, W = X.shape
            if mode == "flatten":
                return X.reshape(N, C * H * W)
            else:  # center pixel
                return X[:, :, H // 2, W // 2]
        return X

    def to(self, device):  # compatibility
        return self

    def train(self):       # compatibility
        return self

    def eval(self):        # compatibility
        return self

class XGBoostModel:
    def __init__(self, params=None):
        """
        Initialize XGBoost model with parameters from config.
        XGBoost equivalent of LightGBM implementation.
        """
        if params is None:
            # Read globals from your existing config parses
            objective        = XGBoost_objective
            eval_metric      = XGBoost_eval_metric
            booster          = XGBoost_booster
            sample_type      = XGBoost_sample_type
            normalize_type   = XGBoost_normalize_type
            rate_drop        = XGBoost_rate_drop
            skip_drop        = XGBoost_skip_drop
            tree_method      = XGBoost_tree_method
            device           = XGBoost_device  # "cpu" or "cuda"
            max_leaves       = int(XGBoost_max_leaves)
            learning_rate    = float(XGBoost_learning_rate)
            colsample_bytree = float(XGBoost_colsample_bytree)
            verbosity        = int(XGBoost_verbosity)

            # Sampling and histogram
            subsample        = float(XGBoost_subsample)
            max_bin          = int(XGBoost_max_bin)
            
            # Regularization
            reg_alpha        = float(XGBoost_reg_alpha)    # L1
            reg_lambda       = float(XGBoost_reg_lambda)   # L2
            min_child_weight = float(XGBoost_min_child_weight)
            gamma            = float(XGBoost_gamma)
            max_depth        = int(XGBoost_max_depth)
            
            # Runtime
            n_jobs           = int(XGBoost_n_jobs)
            
            self.params = {
                "objective": objective,
                "eval_metric": eval_metric,
                "booster": booster,
                "tree_method": tree_method,
                "device": device,
                "sample_type": sample_type,
                "normalize_type": normalize_type,
                "rate_drop": rate_drop,
                "skip_drop": skip_drop,
                "max_leaves": max_leaves,
                "learning_rate": learning_rate,
                "colsample_bytree": colsample_bytree,
                "subsample": subsample,
                "max_bin": max_bin,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "max_depth": max_depth,
                "n_jobs": n_jobs,
                "verbosity": verbosity,
            }
        else:
            self.params = params

        self.model = None
        self.feature_importance = None
        self.best_iteration = 0

    def flatten_features(self, X, mode="center"):
        """
        Convert (N, C, H, W) -> tabular.
        mode='center': use center pixel (fast, your current behavior)
        mode='flatten': use all H*W per channel (may get very wide)
        """
        if X.ndim == 4:
            N, C, H, W = X.shape
            if mode == "flatten":
                return X.reshape(N, C * H * W)
            else:  # center pixel
                return X[:, :, H // 2, W // 2]
        return X

    def to(self, device):  # compatibility
        return self

    def train(self):       # compatibility
        return self

    def eval(self):        # compatibility
        return self

class EnsembleModel:
    """
    Ensemble multiple trained models (LightGBM, CNN, ResNet, etc.)
    Combines predictions using different strategies to improve performance.
    
    Usage:
        # In config.toml, set Ensemble_Settings = true and specify base models
        ensemble = EnsembleModel(base_models=['lightgbm', 'cnn'], strategy='optimized')
        predictions = ensemble.predict(predictions_dict)
    """
    
    def __init__(self, base_models=None, strategy='weighted', weights=None):
        """
        Args:
            base_models: List of model names to ensemble (e.g., ['lightgbm', 'cnn', 'unet'])
            strategy: Ensemble strategy:
                - 'simple': Simple average (equal weights)
                - 'weighted': R²-weighted average (better models get higher weight)
                - 'optimized': Find optimal weights via scipy optimization
                - 'stacking': Use meta-model to learn combination (advanced)
            weights: Custom weights for each model (must sum to 1)
        """
        self.base_models = base_models if base_models is not None else []
        self.strategy = strategy
        self.weights = weights
        self.optimal_weights = None
        self.meta_model = None
        
        # Validate weights if provided
        if self.weights is not None:
            assert len(self.weights) == len(self.base_models), "Number of weights must match number of models"
            assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1"
    
    def simple_average(self, predictions_list):
        """Simple arithmetic mean of all predictions"""
        return np.mean(predictions_list, axis=0)
    
    def weighted_average(self, predictions_list, weights):
        """Weighted average using provided or calculated weights"""
        predictions_array = np.array(predictions_list)  # Shape: (n_models, n_samples)
        weights_array = np.array(weights).reshape(-1, 1)  # Shape: (n_models, 1)
        return np.sum(predictions_array * weights_array, axis=0)
    
    def calculate_r2_weights(self, predictions_list, y_true):
        """
        Calculate weights based on individual model R² scores
        Better performing models get higher weights
        """
        from sklearn.metrics import r2_score
        
        r2_scores = []
        for pred in predictions_list:
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(pred))
            if mask.sum() > 0:
                r2 = r2_score(y_true[mask], pred[mask])
                # Ensure non-negative weights (handle negative R² scores)
                r2_scores.append(max(0.0, r2))
            else:
                r2_scores.append(0.0)
        
        # Normalize to sum to 1
        total_r2 = sum(r2_scores)
        if total_r2 > 0:
            weights = [r2 / total_r2 for r2 in r2_scores]
        else:
            # If all models perform poorly, use equal weights
            weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        return weights
    
    def optimize_weights(self, predictions_list, y_true):
        """
        Find optimal weights using scipy optimization to minimize RMSE
        """
        from scipy.optimize import minimize
        from sklearn.metrics import mean_squared_error
        
        # Remove NaN values
        predictions_array = np.array(predictions_list)  # Shape: (n_models, n_samples)
        mask = ~np.isnan(y_true)
        for pred in predictions_list:
            mask = mask & ~np.isnan(pred)
        
        y_true_clean = y_true[mask]
        predictions_clean = predictions_array[:, mask]
        
        def objective(weights):
            """RMSE as function of weights"""
            ensemble_pred = np.dot(weights, predictions_clean)
            return np.sqrt(mean_squared_error(y_true_clean, ensemble_pred))
        
        # Constraints: weights sum to 1, each weight between 0 and 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(predictions_list))]
        
        # Initial guess: equal weights
        initial_weights = np.ones(len(predictions_list)) / len(predictions_list)
        
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        return result.x
    
    def fit(self, predictions_dict, y_true):
        """
        Fit ensemble model (mainly to find optimal weights)
        
        Args:
            predictions_dict: Dictionary {model_name: predictions_array}
            y_true: True target values
        """
        # Extract predictions in same order as base_models
        predictions_list = [predictions_dict[model] for model in self.base_models]
        
        if self.strategy == 'optimized':
            self.optimal_weights = self.optimize_weights(predictions_list, y_true)
            print(f"Optimal weights found:")
            for model, weight in zip(self.base_models, self.optimal_weights):
                print(f"  {model}: {weight:.4f}")
        
        elif self.strategy == 'weighted' and self.weights is None:
            # Calculate R²-based weights if not provided
            self.weights = self.calculate_r2_weights(predictions_list, y_true)
            print(f"R²-based weights:")
            for model, weight in zip(self.base_models, self.weights):
                print(f"  {model}: {weight:.4f}")
        
        elif self.strategy == 'stacking':
            # Train a simple meta-model (Ridge regression)
            from sklearn.linear_model import Ridge
            
            # Stack predictions as features
            predictions_array = np.array(predictions_list).T  # Shape: (n_samples, n_models)
            
            # Remove NaN values
            mask = ~np.isnan(y_true)
            for pred in predictions_list:
                mask = mask & ~np.isnan(pred)
            
            X_meta = predictions_array[mask]
            y_meta = y_true[mask]
            
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_meta, y_meta)
            
            print(f"Stacking meta-model trained:")
            print(f"  Coefficients: {self.meta_model.coef_}")
            print(f"  Intercept: {self.meta_model.intercept_:.4f}")
    
    def predict(self, predictions_dict):
        """
        Generate ensemble predictions
        
        Args:
            predictions_dict: Dictionary {model_name: predictions_array}
        
        Returns:
            ensemble_predictions: Combined predictions from all models
        """
        # Extract predictions in same order as base_models
        predictions_list = [predictions_dict[model] for model in self.base_models]
        
        if self.strategy == 'simple':
            return self.simple_average(predictions_list)
        
        elif self.strategy == 'weighted':
            weights = self.weights if self.weights is not None else [1.0/len(predictions_list)] * len(predictions_list)
            return self.weighted_average(predictions_list, weights)
        
        elif self.strategy == 'optimized':
            if self.optimal_weights is None:
                raise ValueError("Must call fit() before predict() when using optimized strategy")
            return self.weighted_average(predictions_list, self.optimal_weights)
        
        elif self.strategy == 'stacking':
            if self.meta_model is None:
                raise ValueError("Must call fit() before predict() when using stacking strategy")
            
            # Stack predictions as features
            predictions_array = np.array(predictions_list).T  # Shape: (n_samples, n_models)
            return self.meta_model.predict(predictions_array)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def evaluate(self, predictions_dict, y_true):
        """
        Evaluate ensemble and individual models
        
        Args:
            predictions_dict: Dictionary {model_name: predictions_array}
            y_true: True target values
        
        Returns:
            results_dict: Performance metrics for each model and ensemble
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        results = {}
        
        # Evaluate individual models
        for model_name in self.base_models:
            pred = predictions_dict[model_name]
            mask = ~(np.isnan(y_true) | np.isnan(pred))
            
            if mask.sum() > 0:
                results[model_name] = {
                    'r2': r2_score(y_true[mask], pred[mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true[mask], pred[mask])),
                    'mae': mean_absolute_error(y_true[mask], pred[mask]),
                    'n_samples': mask.sum()
                }
        
        # Evaluate ensemble
        ensemble_pred = self.predict(predictions_dict)
        mask = ~(np.isnan(y_true) | np.isnan(ensemble_pred))
        
        if mask.sum() > 0:
            results['ensemble'] = {
                'r2': r2_score(y_true[mask], ensemble_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], ensemble_pred[mask])),
                'mae': mean_absolute_error(y_true[mask], ensemble_pred[mask]),
                'n_samples': mask.sum(),
                'strategy': self.strategy
            }
        
        return results
    
    # Compatibility methods for your existing pipeline
    def to(self, device):
        """Compatibility with PyTorch models"""
        return self
    
    def train(self):
        """Compatibility with PyTorch models"""
        pass
    
    def eval(self):
        """Compatibility with PyTorch models"""
        pass
