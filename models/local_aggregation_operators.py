import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utlis import create_kernel_points, radius_gaussian, weight_variable
from pt_utils import MaskedQueryAndGroup


class PosPool(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PosPool operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PosPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.position_embedding = config.pospool.position_embedding
        self.reduction = config.pospool.reduction
        self.output_conv = config.pospool.output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)
        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)

        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1) # (B, 1, 3, npoint, nsample)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            raise NotImplementedError(f'Position Embedding {self.position_embedding} not implemented in PosPool')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

"""
The diff. among PosPool, PosPoolSE, ~2 and ~3 are:
- the 1st one is the orginal KPConv version w/o SE idea
- (to add) the 2nd one apply SE idea before the aggregation/max function (**pools over both N and M dim globally** (note: (N,C_out, M))), followed by rescaling/recalibration and aggreatation/max pooling to gain point features
- (to add) the 3rd one apply SE idea before the aggregation/max function (**pools only over M dim globally**),  followed by rescaling/recalibration and aggreatation/max pooling to gain point features
- the 4th one apply SE idea after the aggreation/max pooling function, seems converge faster and achieves better performance. 
"""
class PosPoolSE(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PosPool operator with squeeze-excite idea to recaliberate point feature with global features, which is inspired from SENet(CVPR19) and the paper titled 'Style-based Point Generator with Adversarial Rendering for Point Cloud Completion' (CVPR 2021)

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PosPoolSE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.position_embedding = config.pospool.position_embedding
        self.reduction = config.pospool.reduction
        self.output_conv = config.pospool.output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)
        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

                # nn.Linear(channel, channel // reduction, bias=False),
                # nn.ReLU(inplace=True),
                # nn.Linear(channel // reduction, channel, bias=False),
                # nn.Sigmoid()
        self.excite_tranform = nn.Sequential(
                nn.Conv1d(out_channels, out_channels//16, kernel_size=1, bias=False),
                # nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels//16, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid())

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points. (N2>=N1)
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz,    
                                                query_mask, support_mask, support_features)

        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1) # (B, 1, 3, npoint, nsample)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)

            # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
            # use AdaptiveAvgPool2d?
            global_feature_SE = F.avg_pool2d(aggregation_features,kernel_size=(npoint,self.nsample)).squeeze(-1) # (B,C,1,1)
            global_recalibration_factor=self.excite_tranform(global_feature_SE).unsqueeze(-1) # (B,C,1, 1)
            aggregation_features = aggregation_features * global_recalibration_factor # (B,c,npoint,nsample)

        # TODO: the SE idea is not implemented in this branch
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            raise NotImplementedError(f'Position Embedding {self.position_embedding} not implemented in PosPool')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

class PosPoolSE3(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config, r_SE=16):
        """A PosPool operator with squeeze-excite idea to recaliberate point feature with global features, which is inspired from SENet(CVPR19) and the paper titled 'Style-based Point Generator with Adversarial Rendering for Point Cloud Completion' (CVPR 2021)

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PosPoolSE3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.position_embedding = config.pospool.position_embedding
        self.reduction = config.pospool.reduction
        self.output_conv = config.pospool.output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)
        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

                # nn.Linear(channel, channel // reduction, bias=False),
                # nn.ReLU(inplace=True),
                # nn.Linear(channel // reduction, channel, bias=False),
                # nn.Sigmoid()
        self.excite_tranform = nn.Sequential(
                nn.Conv1d(out_channels, out_channels//r_SE, kernel_size=1, bias=False),
                # nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels//r_SE, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid())

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points. (N2>=N1)
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz,    
                                                query_mask, support_mask, support_features)

        if self.position_embedding == 'xyz':
            position_embedding = torch.unsqueeze(relative_position, 1) # (B, 1, 3, npoint, nsample)
            aggregation_features = neighborhood_features.view(B, C // 3, 3, npoint, self.nsample)
            aggregation_features = position_embedding * aggregation_features  # (B, C//3, 3, npoint, nsample)
            aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)

            # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
            # use AdaptiveAvgPool2d?
            global_feature_SE = F.avg_pool2d(aggregation_features,kernel_size=(npoint,self.nsample)).squeeze(-1) # (B,C,1,1)
            global_recalibration_factor=self.excite_tranform(global_feature_SE).unsqueeze(-1) # (B,C,1, 1)
            aggregation_features = aggregation_features * global_recalibration_factor # (B,c,npoint,nsample)

        # TODO: the SE idea is not implemented in this branch
        elif self.position_embedding == 'sin_cos':
            feat_dim = C // 6
            wave_length = 1000
            alpha = 100
            feat_range = torch.arange(feat_dim, dtype=torch.float32).to(query_xyz.device)  # (feat_dim, )
            dim_mat = torch.pow(1.0 * wave_length, (1.0 / feat_dim) * feat_range)  # (feat_dim, )
            position_mat = torch.unsqueeze(alpha * relative_position, -1)  # (B, 3, npoint, nsample, 1)
            div_mat = torch.div(position_mat, dim_mat)  # (B, 3, npoint, nsample, feat_dim)
            sin_mat = torch.sin(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            cos_mat = torch.cos(div_mat)  # (B, 3, npoint, nsample, feat_dim)
            position_embedding = torch.cat([sin_mat, cos_mat], -1)  # (B, 3, npoint, nsample, 2*feat_dim)
            position_embedding = position_embedding.permute(0, 1, 4, 2, 3).contiguous()
            position_embedding = position_embedding.view(B, C, npoint, self.nsample)  # (B, C, npoint, nsample)
            aggregation_features = neighborhood_features * position_embedding  # (B, C, npoint, nsample)
        else:
            raise NotImplementedError(f'Position Embedding {self.position_embedding} not implemented in PosPool')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1) # (B,C,N)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        # implement SE idea after the pooling sum
        # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
        global_feature_SE = F.avg_pool1d(out_features,kernel_size=(npoint)) # (B,C,1)
        global_recalibration_factor=self.excite_tranform(global_feature_SE) # (B,C,1)
        out_features = out_features * global_recalibration_factor # (B,C, N)


        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features


class AdaptiveWeight(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A AdaptiveWeight operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(AdaptiveWeight, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nsample = nsample
        self.weight_type = config.adaptive_weight.weight_type
        self.weight_to_channels = {'dp': 3,
                                   'df': in_channels,
                                   'fj': in_channels,
                                   'dp_df': 3 + in_channels,
                                   'dp_fj': 3 + in_channels,
                                   'fi_df': 2 * in_channels,
                                   'dp_fi_df': 3 + 2 * in_channels,
                                   'rscnn': 10}
        self.weight_input_channels = self.weight_to_channels[self.weight_type]
        self.num_mlps = config.adaptive_weight.num_mlps
        self.shared_channels = config.adaptive_weight.shared_channels
        self.weight_softmax = config.adaptive_weight.weight_softmax
        self.reduction = config.adaptive_weight.reduction
        self.output_conv = config.adaptive_weight.output_conv or (self.in_channels != self.out_channels)

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)

        self.mlps = nn.Sequential()
        self.mlps.add_module('conv0',
                             nn.Conv2d(self.weight_input_channels,
                                       self.in_channels // self.shared_channels,
                                       kernel_size=1))
        for i in range(self.num_mlps - 1):
            self.mlps.add_module(f'relu{i}', nn.ReLU(inplace=True))
            self.mlps.add_module(f'conv{i + 1}',
                                 nn.Conv2d(self.in_channels // self.shared_channels,
                                           self.in_channels // self.shared_channels,
                                           kernel_size=1))

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)

        if self.weight_type == 'dp':
            conv_weight = self.mlps(relative_position)  # (B, C//S, npoint, nsample)
            conv_weight = torch.unsqueeze(conv_weight, 2)  # (B, C//S, 1, npoint, nsample)
        else:
            raise NotImplementedError(f'Weight Type {self.weight_type} not implemented in AdaptiveWeight')

        aggregation_features = neighborhood_features.view(B, C // self.shared_channels, self.shared_channels,
                                                          npoint, self.nsample)
        aggregation_features = aggregation_features * conv_weight
        aggregation_features = aggregation_features.view(B, C, npoint, self.nsample)

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample]
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PosPool ')

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features


class PointWiseMLP(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PointWiseMLP operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PointWiseMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nsample = nsample
        self.feature_type = config.pointwisemlp.feature_type
        self.feature_input_channels = {'dp_fj': 3 + in_channels,
                                       'fi_df': 2 * in_channels,
                                       'dp_fi_df': 3 + 2 * in_channels}
        self.feature_input_channels = self.feature_input_channels[self.feature_type]
        self.num_mlps = config.pointwisemlp.num_mlps
        self.reduction = config.pointwisemlp.reduction

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)

        self.mlps = nn.Sequential()
        if self.num_mlps == 1:
            self.mlps.add_module('conv0', nn.Sequential(
                nn.Conv2d(self.feature_input_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))
        else:
            mfdim = max(self.in_channels // 2, 9)
            self.mlps.add_module('conv0', nn.Sequential(
                nn.Conv2d(self.feature_input_channels, mfdim, kernel_size=1, bias=False),
                nn.BatchNorm2d(mfdim, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))
            for i in range(self.num_mlps - 2):
                self.mlps.add_module(f'conv{i + 1}', nn.Sequential(
                    nn.Conv2d(mfdim, mfdim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mfdim, momentum=config.bn_momentum),
                    nn.ReLU(inplace=True)))
            self.mlps.add_module(f'conv{self.num_mlps - 1}', nn.Sequential(
                nn.Conv2d(mfdim, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        if self.feature_type == 'dp_fi_df':
            # B C N M
            center_features = torch.unsqueeze(neighborhood_features[..., 0], -1).repeat([1, 1, 1, self.nsample])
            relative_features = neighborhood_features - center_features
            local_input_features = torch.cat([relative_position, center_features, relative_features], 1)
            aggregation_features = self.mlps(local_input_features) # (B,C_out,N1,nsample)
        else:
            raise NotImplementedError(f'Feature Type {self.feature_type} not implemented in PointWiseMLP')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample] # (B,C_out, N1)
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None]) # (B,N1=15000,nsample) + (B,N1,1)
            # why feature_mask computes like this? each padded pt's nb mask will get 1, or 2?
            feature_mask = feature_mask[:, None, :, :] # (B,1,N1,nsample)
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PointWiseMLP')
        return out_features


"""
The diff. among PseudoGrid, PseudoGridSE, ~2 and ~3 are:
- the 1st one is the orginal KPConv version w/o SE idea
- the 2nd one apply SE idea before the aggregation/max function (**pools over both N and M dim globally** (note: (N,C_out, M))), followed by rescaling/recalibration and aggreatation/max pooling to gain point features
- the 3rd one apply SE idea before the aggregation/max function (**pools only over M dim globally**),  followed by rescaling/recalibration and aggreatation/max pooling to gain point features
- the 4th one apply SE idea after the aggreation/max pooling function, seems converge faster and achieves better performance. 
"""
class PseudoGrid(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PseudoGrid operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PseudoGrid, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.KP_influence = config.pseudo_grid.KP_influence
        self.num_kernel_points = config.pseudo_grid.num_kernel_points
        self.convolution_mode = config.pseudo_grid.convolution_mode
        self.output_conv = config.pseudo_grid.output_conv or (self.in_channels != self.out_channels)

        # create kernel points
        KP_extent = config.pseudo_grid.KP_extent
        fixed_kernel_points = config.pseudo_grid.fixed_kernel_points
        density_parameter = config.density_parameter
        self.extent = 2 * KP_extent * radius / density_parameter
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(K_radius,
                                              self.num_kernel_points,
                                              num_kernels=1,
                                              dimension=3,
                                              fixed=fixed_kernel_points)

        K_points_numpy = K_points_numpy.reshape((self.num_kernel_points, 3))
        self.register_buffer('K_points', torch.from_numpy(K_points_numpy).type(torch.float32))

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=False)
        self.kernel_weights = weight_variable([self.num_kernel_points, in_channels])

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        # (B,C,N,K)
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        relative_position = torch.unsqueeze(relative_position.permute(0, 2, 3, 1), 3)
        relative_position = relative_position.repeat([1, 1, 1, self.num_kernel_points, 1])

        # Get Kernel point influences [B, N, K, M]
        differences = relative_position - self.K_points
        sq_distances = torch.sum(differences ** 2, -1)
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.extent, min=0.0)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = all_weights.permute(0, 1, 3, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # Mask padding points
        feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])  # B, N, M
        all_weights *= feature_mask[:, :, None, :]  # B, N, K, M

        if self.convolution_mode != 'sum':
            raise NotImplementedError(f"convolution_mode:{self.convolution_mode} not support in PseudoGrid")

        # get features for each kernel point
        all_weights = all_weights.view(-1, self.num_kernel_points, self.nsample)
        neighborhood_features = neighborhood_features.permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, C)
        weighted_features = torch.bmm(all_weights, neighborhood_features)  # # [B*N, K, M],[B*N, M, C] -> [B*N, K, C]
        kernel_outputs = weighted_features * self.kernel_weights  # [B*N, K, C]
        out_features = torch.sum(kernel_outputs, 1).view(B, npoint, C).transpose(1, 2)

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

class PseudoGridSE(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PseudoGrid operator with squeeze and excite idea for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PseudoGridSE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.KP_influence = config.pseudo_grid.KP_influence
        self.num_kernel_points = config.pseudo_grid.num_kernel_points
        self.convolution_mode = config.pseudo_grid.convolution_mode
        self.output_conv = config.pseudo_grid.output_conv or (self.in_channels != self.out_channels)

        # create kernel points
        KP_extent = config.pseudo_grid.KP_extent
        fixed_kernel_points = config.pseudo_grid.fixed_kernel_points
        density_parameter = config.density_parameter
        self.extent = 2 * KP_extent * radius / density_parameter
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(K_radius,
                                              self.num_kernel_points,
                                              num_kernels=1,
                                              dimension=3,
                                              fixed=fixed_kernel_points)

        K_points_numpy = K_points_numpy.reshape((self.num_kernel_points, 3))
        self.register_buffer('K_points', torch.from_numpy(K_points_numpy).type(torch.float32))

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=False)
        self.kernel_weights = weight_variable([self.num_kernel_points, in_channels])

        # for implement SE idea
                # nn.Linear(channel, channel // reduction, bias=False),
                # nn.ReLU(inplace=True),
                # nn.Linear(channel // reduction, channel, bias=False),
                # nn.Sigmoid()
        self.excite_tranform = nn.Sequential(
                nn.Conv1d(out_channels, out_channels//16, kernel_size=1, bias=False),
                # nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels//16, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid())

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        # (B,C,N,K)
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        relative_position = torch.unsqueeze(relative_position.permute(0, 2, 3, 1), 3)
        relative_position = relative_position.repeat([1, 1, 1, self.num_kernel_points, 1])

        # Get Kernel point influences [B, N, K, M]
        differences = relative_position - self.K_points
        sq_distances = torch.sum(differences ** 2, -1)
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.extent, min=0.0)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = all_weights.permute(0, 1, 3, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # Mask padding points
        feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])  # B, N, M
        all_weights *= feature_mask[:, :, None, :]  # B, N, K, M

        if self.convolution_mode != 'sum':
            raise NotImplementedError(f"convolution_mode:{self.convolution_mode} not support in PseudoGrid")

        # get features for each kernel point
        all_weights = all_weights.view(-1, self.num_kernel_points, self.nsample) # (B*N, K=20, M=15)
        neighborhood_features = neighborhood_features.permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, C) # (B*N,M,C)
        weighted_features = torch.bmm(all_weights, neighborhood_features)  # [B*N, K, M],[B*N, M, C] -> [B*N, K, C]
        kernel_outputs = weighted_features * self.kernel_weights  # [B*N, K, C]

        # implement SE idea
        aggregation_features = kernel_outputs.view(B, npoint, -1, C).permute(0,3,1,2).contiguous()  # (B, C, npoint, M=15)
        # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
        # use AdaptiveAvgPool2d?
        global_feature_SE = F.avg_pool2d(aggregation_features,kernel_size=(npoint,self.num_kernel_points)).squeeze(-1) # (B,C,1,1)
        global_recalibration_factor=self.excite_tranform(global_feature_SE).unsqueeze(-1) # (B,C,1, 1)
        aggregation_features = aggregation_features * global_recalibration_factor # (B,C,npoint, K)
        kernel_outputs = aggregation_features.permute(0,2,3,1).contiguous().view(-1,self.num_kernel_points,C) # (B*npoint,K,C)

        # aggregation over the neighborhood
        out_features = torch.sum(kernel_outputs, 1).view(B, npoint, C).transpose(1, 2) # (B, C, N)
        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

class PseudoGridSE2(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A PseudoGrid operator with squeeze and excite idea for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PseudoGridSE2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.KP_influence = config.pseudo_grid.KP_influence
        self.num_kernel_points = config.pseudo_grid.num_kernel_points
        self.convolution_mode = config.pseudo_grid.convolution_mode
        self.output_conv = config.pseudo_grid.output_conv or (self.in_channels != self.out_channels)

        # create kernel points
        KP_extent = config.pseudo_grid.KP_extent
        fixed_kernel_points = config.pseudo_grid.fixed_kernel_points
        density_parameter = config.density_parameter
        self.extent = 2 * KP_extent * radius / density_parameter
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(K_radius,
                                              self.num_kernel_points,
                                              num_kernels=1,
                                              dimension=3,
                                              fixed=fixed_kernel_points)

        K_points_numpy = K_points_numpy.reshape((self.num_kernel_points, 3))
        self.register_buffer('K_points', torch.from_numpy(K_points_numpy).type(torch.float32))

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=False)
        self.kernel_weights = weight_variable([self.num_kernel_points, in_channels])

        # for implement SE idea
                # nn.Linear(channel, channel // reduction, bias=False),
                # nn.ReLU(inplace=True),
                # nn.Linear(channel // reduction, channel, bias=False),
                # nn.Sigmoid()
        self.excite_tranform = nn.Sequential(
                nn.Conv1d(out_channels, out_channels//16, kernel_size=1, bias=False),
                # nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels//16, out_channels, kernel_size=1, bias=False),
                nn.Sigmoid())

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        # (B,C,N,K)
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        relative_position = torch.unsqueeze(relative_position.permute(0, 2, 3, 1), 3)
        relative_position = relative_position.repeat([1, 1, 1, self.num_kernel_points, 1])

        # Get Kernel point influences [B, N, K, M]
        differences = relative_position - self.K_points
        sq_distances = torch.sum(differences ** 2, -1)
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.extent, min=0.0)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = all_weights.permute(0, 1, 3, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # Mask padding points
        feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])  # B, N, M
        all_weights *= feature_mask[:, :, None, :]  # B, N, K, M

        if self.convolution_mode != 'sum':
            raise NotImplementedError(f"convolution_mode:{self.convolution_mode} not support in PseudoGrid")

        # get features for each kernel point
        all_weights = all_weights.view(-1, self.num_kernel_points, self.nsample) # (B*N, K=20, M=15)
        neighborhood_features = neighborhood_features.permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, C) # (B*N,M,C)
        weighted_features = torch.bmm(all_weights, neighborhood_features)  # [B*N, K, M],[B*N, M, C] -> [B*N, K, C]
        kernel_outputs = weighted_features * self.kernel_weights  # [B*N, K, C]

        # implement SE idea
        aggregation_features = kernel_outputs.view(B, npoint, -1, C).permute(0,3,1,2).contiguous()  # (B, C, npoint, M=15)
        # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
        # use AdaptiveAvgPool2d?
        global_feature_SE = F.avg_pool2d(aggregation_features,kernel_size=(1,self.num_kernel_points)).squeeze(-1) # (B,C,N,1)
        global_recalibration_factor=self.excite_tranform(global_feature_SE).unsqueeze(-1) # (B,C,N, 1)
        aggregation_features = aggregation_features * global_recalibration_factor # (B,C, N, M=15)
        kernel_outputs = aggregation_features.permute(0,2,3,1).contiguous().view(-1,self.num_kernel_points,C) # (B*npoint,K,C)

        # aggregation over the neighborhood
        out_features = torch.sum(kernel_outputs, 1).view(B, npoint, C).transpose(1, 2) # (B, C, N)
        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

class PseudoGridSE3(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config, r_SE=16, 
        squeeze_type='avg', excitation_type='sigmoid'):
        """A PseudoGrid operator with squeeze and excite idea for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(PseudoGridSE3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.nsample = nsample
        self.KP_influence = config.pseudo_grid.KP_influence
        self.num_kernel_points = config.pseudo_grid.num_kernel_points
        self.convolution_mode = config.pseudo_grid.convolution_mode
        self.output_conv = config.pseudo_grid.output_conv or (self.in_channels != self.out_channels)

        # squeeze and exictation type
        if config.SE_squeeze_type:
            self.SE_squeeze_type=config.SE_squeeze_type # avg or max
        if config.SE_excitation_type:
            self.SE_excitation_type=config.SE_excitation_type # sigmoid, relu or tanh

        # create kernel points
        KP_extent = config.pseudo_grid.KP_extent
        fixed_kernel_points = config.pseudo_grid.fixed_kernel_points
        density_parameter = config.density_parameter
        self.extent = 2 * KP_extent * radius / density_parameter
        K_radius = 1.5 * self.extent
        K_points_numpy = create_kernel_points(K_radius,
                                              self.num_kernel_points,
                                              num_kernels=1,
                                              dimension=3,
                                              fixed=fixed_kernel_points)

        K_points_numpy = K_points_numpy.reshape((self.num_kernel_points, 3))
        self.register_buffer('K_points', torch.from_numpy(K_points_numpy).type(torch.float32))

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=False)
        self.kernel_weights = weight_variable([self.num_kernel_points, in_channels])

        # for implement SE idea
                # nn.Linear(channel, channel // reduction, bias=False),
                # nn.ReLU(inplace=True),
                # nn.Linear(channel // reduction, channel, bias=False),
                # nn.Sigmoid()
        if self.SE_excitation_type == 'relu':
            activation_function=nn.ReLU
        elif self.SE_excitation_type == 'tanh':
            activation_function=nn.Tanh
        else:
            activation_function=nn.Sigmoid

        self.excite_tranform = nn.Sequential(
                nn.Conv1d(out_channels, out_channels//r_SE, kernel_size=1, bias=False),
                # nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels//r_SE, out_channels, kernel_size=1, bias=False),
                activation_function())

        if self.output_conv:
            self.out_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))
        else:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        B = query_xyz.shape[0]
        C = support_features.shape[1]
        npoint = query_xyz.shape[1]
        # (B,C,N,K)
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        relative_position = torch.unsqueeze(relative_position.permute(0, 2, 3, 1), 3)
        relative_position = relative_position.repeat([1, 1, 1, self.num_kernel_points, 1])

        # Get Kernel point influences [B, N, K, M]
        differences = relative_position - self.K_points
        sq_distances = torch.sum(differences ** 2, -1)
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.extent, min=0.0)
            all_weights = all_weights.permute(0, 1, 3, 2)
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = all_weights.permute(0, 1, 3, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # Mask padding points
        feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])  # B, N, M
        all_weights *= feature_mask[:, :, None, :]  # B, N, K, M

        if self.convolution_mode != 'sum':
            raise NotImplementedError(f"convolution_mode:{self.convolution_mode} not support in PseudoGrid")

        # get features for each kernel point
        all_weights = all_weights.view(-1, self.num_kernel_points, self.nsample) # (B*N, K=20, M=15)
        neighborhood_features = neighborhood_features.permute(0, 2, 3, 1).contiguous().view(-1, self.nsample, C) # (B*N,M,C)
        weighted_features = torch.bmm(all_weights, neighborhood_features)  # [B*N, K, M],[B*N, M, C] -> [B*N, K, C]
        kernel_outputs = weighted_features * self.kernel_weights  # [B*N, K, C]

        # aggregation over the neighborhood
        out_features = torch.sum(kernel_outputs, 1).view(B, npoint, C).transpose(1, 2) # (B, C, N)

        # implement SE idea after the pooling sum
        # apply SE idea: 1)squeeze and excite the feature map(aggregation_features), 2)mulitplied with the aggregation features
        if self.SE_squeeze_type == 'avg':
            global_feature_SE = F.avg_pool1d(out_features,kernel_size=(npoint)) # (B,C,1)
        else:
            global_feature_SE = F.max_pool1d(out_features,kernel_size=(npoint)) # (B,C,1)
        global_recalibration_factor=self.excite_tranform(global_feature_SE) # (B,C,1)
        out_features = out_features * global_recalibration_factor # (B,C, N)

        if self.output_conv:
            out_features = self.out_conv(out_features)
        else:
            out_features = self.out_transform(out_features)

        return out_features

class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """LocalAggregation operators

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(LocalAggregation, self).__init__()
        if config.local_aggregation_type == 'pospool':
            self.local_aggregation_operator = PosPool(in_channels, out_channels, radius, nsample, config)
        # the 2nd one apply SE idea before the aggregation/max function (**pools over both N and M dim globally** (note: (N,C_out, M))), followed by rescaling/recalibration and aggreatation/max pooling to gain point features
        elif config.local_aggregation_type == 'pospool_se':
            self.local_aggregation_operator = PosPoolSE(in_channels, out_channels, radius, nsample, config)
        # the 4th one apply SE idea after the aggreation/max pooling function, seems converge faster and achieves better performance. 
        elif config.local_aggregation_type == 'pospool_se3':
            self.local_aggregation_operator = PosPoolSE3(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'adaptive_weight':
            self.local_aggregation_operator = AdaptiveWeight(in_channels, out_channels, radius, nsample, config)
        elif config.local_aggregation_type == 'pointwisemlp':
            self.local_aggregation_operator = PointWiseMLP(in_channels, out_channels, radius, nsample, config)
        # original pseudo grid
        elif config.local_aggregation_type == 'pseudo_grid':
            self.local_aggregation_operator = PseudoGrid(in_channels, out_channels, radius, nsample, config)
        # the 2nd one apply SE idea before the aggregation/max function (**pools over both N and M dim globally** (note: (N,C_out, M))), followed by rescaling/recalibration and aggreatation/max pooling to gain point features
        elif config.local_aggregation_type == 'pseudo_grid_se':
            self.local_aggregation_operator = PseudoGridSE(in_channels, out_channels, radius, nsample, config)
        # the 3rd one apply SE idea before the aggregation/max function (**pools only over M dim globally**),  followed by rescaling/recalibration and aggreatation/max pooling to gain point features
        elif config.local_aggregation_type == 'pseudo_grid_se2':
            self.local_aggregation_operator = PseudoGridSE2(in_channels, out_channels, radius, nsample, config)
        # the 4th one apply SE idea after the aggreation/max pooling function, seems converge faster and achieves better performance. 
        elif config.local_aggregation_type == 'pseudo_grid_se3':
            self.local_aggregation_operator = PseudoGridSE3(in_channels, out_channels, radius, nsample, config)
        else:
            raise NotImplementedError(f'LocalAggregation {config.local_aggregation_type} not implemented')

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.local_aggregation_operator(query_xyz, support_xyz, query_mask, support_mask, support_features)
