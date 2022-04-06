import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalAvgPool1d, self).__init__()

    def forward(self, mask, features):
        """[added by yc]

        Args:
            mask ([type]): mask, BxN, each value is 0 or 1.
            features ([type]): BxC'xN

        Returns:
            [type]: [description]
        """
        out = features.sum(-1) # BxC'
        pcl_num = mask.sum(-1) # B,
        out /= pcl_num[:, None] # BxC'
        return out # BxC'

class MaskedGlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalMaxPool1d, self).__init__()

    def forward(self, mask, features):
        """[added by yc]

        Args:
            mask ([type]): mask, BxC'xN, each value is 0 or 1.
            features ([type]): BxC'xN

        Returns:
            [type]: [description]
        """
        out,_ = features.max(-1) # BxC'
        # out = torch.unsqueeze(out,dim=-1)
        return out # BxC'x1


class ClassifierResNet(nn.Module):
    def __init__(self, num_classes, width, use_avg_max_pool=False):
        """A classifier for ResNet backbone.

        Args:
            num_classes: the number of classes.
            width: the base channel num.

        Returns:
            logits: (B, num_classes)
        """
        super(ClassifierResNet, self).__init__()
        self.num_classes = num_classes
        self.pool_avg = MaskedGlobalAvgPool1d()
        self.pool_max = MaskedGlobalMaxPool1d()
        # whether use default pospool setting(only global avg pooling) or global avg+max pooling--yc
        self.use_avg_max_pool= use_avg_max_pool
        if self.use_avg_max_pool:
            # concat avg and max feaures will double
            width*=2
        self.classifier = nn.Sequential(
            nn.Linear(16 * width, 8 * width),
            nn.BatchNorm1d(8 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8 * width, 4 * width),
            nn.BatchNorm1d(4 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4 * width, 2 * width),
            nn.BatchNorm1d(2 * width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2 * width, num_classes))


    def forward(self, end_points):
        if self.use_avg_max_pool:
            pooled_features_avg = self.pool_avg(end_points['res5_mask'], end_points['res5_features'])
            pooled_features_max = self.pool_max(end_points['res5_mask'], end_points['res5_features'])
            # x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
            pooled_features=torch.cat((pooled_features_max,pooled_features_avg),dim=1) # Bx2Cx1
        else:
            pooled_features = self.pool_avg(end_points['res5_mask'], end_points['res5_features'])

        return self.classifier(pooled_features)
