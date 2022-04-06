import numpy as np
import torch
import torch.nn as nn

from .backbones import ResNet
from .heads import ClassifierResNet, MultiPartSegHeadResNet, SceneSegHeadResNet
from .losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy


def build_classification(config):
    model = ClassificationModel(config,
                                config.backbone, config.head, config.num_classes, config.input_features_dim,
                                config.radius, config.sampleDl, config.nsamples, config.npoints,
                                config.width, config.depth, config.bottleneck_ratio)
    if 'ce' in config.loss:
        if config.data_freq:
            # inverse frequence, refer to SensatUrban paper
            if config.loss=='wce':
                weight = torch.tensor(1/np.array(config.data_freq)).float()
            # inverse square root frequence
            elif config.loss=='sqrt_ce':
                weight = torch.tensor(1/np.sqrt(np.array(config.data_freq))).float()
            else:
                raise ValueError('Unknown loss type, use either wce or sqrt_ce loss')
        else:
            # no weighted for the CE loss
            weight = None

        print('loss weights are {}\n'.format(weight))
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif config.loss=='smooth':
        criterion = LabelSmoothingCrossEntropyLoss()
    else:
        raise ValueError('Unknown loss type, use either ce or smooth')

    return model, criterion


def build_multi_part_segmentation(config):
    model = MultiPartSegmentationModel(config, config.backbone, config.head, config.num_classes, config.num_parts,
                                       config.input_features_dim,
                                       config.radius, config.sampleDl, config.nsamples, config.npoints,
                                       config.width, config.depth, config.bottleneck_ratio)
    criterion = MultiShapeCrossEntropy(config.num_classes)
    return model, criterion


def build_scene_segmentation(config):
    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    criterion = MaskedCrossEntropy()
    return model, criterion


class ClassificationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(ClassificationModel, self).__init__()

        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Classification Model")

        if head == 'resnet_cls':
            self.classifier = ClassifierResNet(num_classes, width, config.use_avg_max_pool)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Classification Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.classifier(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class MultiPartSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, num_parts,
                 input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(MultiPartSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_part_seg':
            self.segmentation_head = MultiPartSegHeadResNet(num_classes, width, radius, nsamples, num_parts)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class SceneSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        super(SceneSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_scene_seg':
            self.segmentation_head = SceneSegHeadResNet(num_classes, width, radius, nsamples)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
