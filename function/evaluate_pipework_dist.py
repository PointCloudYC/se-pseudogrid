"""
Distributed evaluating script for 3D shape classification with PipeWork dataset
"""
import argparse
import os
import sys
import time
import json
import random
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

from models import build_classification
from datasets import PipeWorkCls
import datasets.data_utils as d_utils
from utils.util import AverageMeter, accuracy, str2bool, classification_metrics, plot_CM_wrapper, plot_wrongly_predicted_point_clouds, save_fig, fashion_scatter
from utils.logger import setup_logger
from utils.config import config, update_config


def parse_option():
    parser = argparse.ArgumentParser('PipeWork classification evaluating')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--load_path', required=True, type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--loss', type=str, default='smooth', help='loss types, e.g., smooth or ce or wce or sqrt_ce')
    parser.add_argument("--use_avg_max_pool", type=str2bool, default=False, help='whether to apply avg and max pooling globally for the classification, need concat them.')
    parser.add_argument('--log_dir', type=str, default='log_eval', help='log dir [default: log_eval]')
    parser.add_argument('--data_root', type=str, default='data', help='root director of dataset')
    parser.add_argument("--data_aug", type=str2bool, default=True, help='whether to apply data augmentation')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--num_points', type=int, help='num_points')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')

    # SE module
    parser.add_argument('--SE_squeeze_type', type=str, default='avg', help='squeeze types for SE, e.g., avg or max')
    parser.add_argument('--SE_excitation_type', type=str, default='sigmoid', help='excitation types for SE, e.g., sigmoid, relu or tanh')

    # plot t-SNE figure; Note: the t-SNE figure is not very good (less separated) compared w. the effect of t-SNE on MINIST 
    parser.add_argument("--tsne", type=str2bool, default=False, help='whether to plot t-SNE figure on learned global features')

    args, unparsed = parser.parse_known_args()
    update_config(args.cfg)

    config.data_root = args.data_root
    config.use_avg_max_pool = args.use_avg_max_pool
    config.loss = args.loss
    config.data_aug = args.data_aug
    config.num_workers = args.num_workers
    config.load_path = args.load_path
    config.rng_seed = args.rng_seed
    config.local_rank = args.local_rank
    # SE module
    config.SE_squeeze_type = args.SE_squeeze_type
    config.SE_excitation_type = args.SE_excitation_type
    # t-SNE figure
    config.tsne = args.tsne

    ddir_name = args.cfg.split('.')[-2].split('/')[-1]
    # Note: different folder name from the training log (i.e., ckpt folder for training)
    if config.data_aug:
        config.log_dir = os.path.join(args.log_dir, 'pipework', f'{ddir_name}_{int(time.time())}_DA')
    else:
        config.log_dir = os.path.join(args.log_dir, 'pipework', f'{ddir_name}_{int(time.time())}_no_DA')
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_points:
        config.num_points = args.num_points

    print(args)
    print(config)

    # torch.manual_seed(args.rng_seed)
    # torch.cuda.manual_seed_all(args.rng_seed)
    # random.seed(args.rng_seed)
    # np.random.seed(args.rng_seed)

    return args, config


def get_loader(args):
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    test_dataset = PipeWorkCls(input_features_dim=config.input_features_dim, num_points=args.num_points,
                                 data_root=args.data_root, transforms=test_transforms,
                                 subsampling_parameter=config.sampleDl,
                                 split='test')

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)

    return test_loader


def load_checkpoint(config, model):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location='cpu')
    config.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def main(config,path=None):
    test_loader = get_loader(config)
    n_data = len(test_loader.dataset) # 934 instances in the test set
    logger.info(f"length of testing dataset: {n_data}")

    model, criterion = build_classification(config) # use a label smoothing CE loss
    model.cuda()
    criterion.cuda()

    model = DistributedDataParallel(model, device_ids=[config.local_rank], broadcast_buffers=False)

    # resume from a checkpoint to validate
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(config, model)
        logger.info("==> checking loaded ckpt")
        validate(test_loader, model, criterion, config, path, num_votes=10)

def get_avg_global_features(points, mask, features, model):
    """obtain the avg pooling global features
    Args:
        points ([type]): points_batch
        mask ([type]): points mask
        features ([type]): point features
        model ([type]): the loaded model
    Returns:
        [type]: return the avg global features
    """
    with torch.no_grad():
        output = None
        
        def hook_func(module_, input_, output_):
            nonlocal output
            output = output_

        # the model's name(pool_avg) is determined by the classifer definition
        hook = model.module.classifier.pool_avg.register_forward_hook(hook_func)        
        model(points, mask, features) # (B,num_classes)
        hook.remove()

        return output

def get_max_global_features(points, mask, features, model):
    """obtain the max pooling global features
    Args:
        points ([type]): points_batch
        mask ([type]): points mask
        features ([type]): point features
        model ([type]): the loaded model
    Returns:
        [type]: return the avg global features
    """
    with torch.no_grad():
        output = None
        
        def hook_func(module_, input_, output_):
            nonlocal output
            output = output_

        # the model's name(pool_avg) is determined by the classifer definition
        hook = model.module.classifier.pool_max.register_forward_hook(hook_func)        
        model(points, mask, features) # (B,num_classes)
        hook.remove()

        return output

def validate(test_loader, model, criterion, config, path=None, num_votes=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        vote_preds = None
        TS = d_utils.BatchPointcloudScaleAndJitter(scale_low=config.scale_low,
                                                   scale_high=config.scale_high,
                                                   std=config.noise_std,
                                                   clip=config.noise_clip)
        for v in range(num_votes):
            preds = []
            targets = []
            global_features=[]
            for idx, (points, mask, features, target) in enumerate(test_loader):
                # augment for voting
                if v > 0 and config.data_aug:
                    points = TS(points)
                    if config.input_features_dim == 3:
                        features = points
                        features = features.transpose(1, 2).contiguous()
                    elif config.input_features_dim == 4:
                        features = torch.ones(size=(points.shape[0], points.shape[1], 1), dtype=torch.float32)
                        features = torch.cat([features, points], -1)
                        features = features.transpose(1, 2).contiguous()
                    else:
                        raise NotImplementedError(
                            f"input_features_dim {config.input_features_dim} in voting not supported")

                # forward
                points = points.cuda(non_blocking=True) # (B,N,3)
                mask = mask.cuda(non_blocking=True) # (B,N)
                features = features.cuda(non_blocking=True) # (B,3,N)
                target = target.cuda(non_blocking=True) # (B,)

                # when t-sne, then collect global features (either avg or both)
                if config.tsne:
                    # obtained global features
                    global_feature_avg = get_avg_global_features(points, mask, features, model)
                    if config.use_avg_max_pool:
                        global_feature_max = get_avg_global_features(points, mask, features, model)
                        global_feature = torch.cat((global_feature_max, global_feature_avg),dim=1) # Bx2Cx1
                    else:
                        global_feature = global_feature_avg
                    global_features.append(global_feature)

                pred = model(points, mask, features) # (B,num_classes)
                target = target.view(-1)
                loss = criterion(pred, target)
                acc1 = accuracy(pred, target, topk=(1,)) # array
                # no need to compute average accuracy for each batch since many category acc are 0.
                # acc, avg_class_acc = classification_metrics(pred.cpu().numpy(), target.cpu().numpy(), num_classes=config.num_classes)

                losses.update(loss.item(), points.size(0))
                top1.update(acc1[0].item(), points.size(0))

                preds.append(pred)
                targets.append(target)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if idx % config.print_freq == 0:
                    logger.info(
                        f'Test: [{idx}/{len(test_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})')
            logger.info(f' * Acc@1 {top1.avg:.3%}')
            top1.reset()

            preds = torch.cat(preds, 0)
            targets = torch.cat(targets, 0)

            if vote_preds is None:
                vote_preds = preds
            else:
                # sum all logits for voting predictions
                vote_preds += preds 
            vote_acc1 = accuracy(vote_preds, targets, topk=(1,))[0].item()
            logger.info(f' * Vote{v} Acc@1 {vote_acc1:.3%}')
            _, vote_avg_acc = classification_metrics(vote_preds.cpu().numpy(), targets.cpu().numpy(), num_classes=config.num_classes)
            logger.info(f' * Vote{v} avg acc {vote_avg_acc:.3%}')

            # ouput more eval metrics(precision, recall, etc) and confusion matrix in the last voting
            if v==num_votes-1:
                # precision, recall, f1-score, etc.
                logger.info(f' * More evaluation metrics of Vote{v}:')
                label_to_names = {0: 'BlindFlange', 1: 'Cross', 2: 'Elbow90', 3: 'Elbownon90', 4: 'Flange', 5: 'FlangeWN', 6: 'Olet', 7: 'OrificeFlange', 8: 'Pipe', 9: 'ReducerCONC', 10: 'ReducerECC', 11: 'ReducerInsert', 12: 'SafetyValve', 13: 'Strainer', 14: 'Tee', 15: 'TeeRED', 16: 'Valve'}
                target_names = list(label_to_names.values())
                y_true = targets.cpu().numpy()
                y_pred = np.argmax(vote_preds.cpu().numpy(), -1)
                cls_report = classification_report(
                        y_true,
                        y_pred,
                        target_names=target_names,
                        digits=4)
                logger.info(f'\n{cls_report}')

                """ 
                # draw t-SNE figure for the global features
                # the generated figure is not good as t-SNE figure is not very well sperated as the MINIST t-SNE effect
                # still the legend can not be generated, see plot/images for results
                """
                if config.tsne:
                    if path:
                        # path log_eval/pipework/pseudo_grid_1629165562/config.json
                        save_path = os.path.join(*path.split('/')[:-1])
                    else:
                        save_path = os.path.join('images')
                    global_features = torch.cat(global_features, 0) # (N, 4608 =2304*2)
                    # dump the global features
                    filename = os.path.join(save_path, f'global_features_test.pkl')
                    with open(filename, 'wb') as f:
                        pickle.dump((global_features.cpu().numpy(), targets.cpu().numpy()), f)

                    # plot t-SNE for all categories
                    time_start = time.time()
                    pipework_tsne = TSNE(random_state=123).fit_transform(global_features.cpu().numpy())
                    logger.info('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
                    fashion_scatter(pipework_tsne, targets.cpu().numpy())
                    save_fig(save_path,f'tSNE_pipework', tight_layout=True)

                    # plot t-SNE for selected categories
                    # Frequent classes: [0,2,3,4,5,9,14,15,16]
                    # Misclassified classes: [2,3,4,5,9,10,14,15]
                    frequent_classes = [0,2,3,4,5,9,14,15,16]
                    mask_all = np.zeros(targets.shape[0]).astype(bool)
                    for item in frequent_classes:
                        mask_all = mask_all | (targets.cpu().numpy()==item)
                    global_features_filtered = global_features.cpu().numpy()[mask_all] # [N'=873, 4608]
                    targets_filtered = targets.cpu().numpy()[mask_all] # [N'=873]
                    time_start = time.time()
                    pipework_tsne = TSNE(random_state=123).fit_transform(global_features_filtered)
                    logger.info('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
                    fashion_scatter(pipework_tsne, targets_filtered, True)
                    save_fig(save_path,f'tSNE_pipework_filtered', tight_layout=True)

                # plot confusion matrix
                if path:
                    # path log_eval/pipework/pseudo_grid_1629165562/config.json
                    save_path = os.path.join(*path.split('/')[:-1])
                    C = confusion_matrix(
                            y_true, 
                            y_pred, 
                            np.arange(config.num_classes)) 
                    # plot 3 figures, 1 default style and seaborn style w or w/o percents
                    plot_CM_wrapper(
                        C,y_true,y_pred,
                        label_to_names,save_path,
                        filename='CM_seaborn',
                        figsize=(12,12),
                        fmt='0.1f')
                    logger.info("Confusion matrix saved to {}".format(save_path))

                    # plot wrongly predicted examples for error analysis
                    indices_wrong = plot_wrongly_predicted_point_clouds(
                        y_true, 
                        y_pred, 
                        test_loader,
                        save_path,
                        label_to_names,
                        filename='wrongly_predicted_point_clouds',
                        sampling_ratio=0.1)
                    logger.info("{} wrong predictions!".format(len(indices_wrong)))
                    logger.info("The indices of test set which are predicted wrongly are: {}".format(indices_wrong))

    return vote_acc1, vote_avg_acc


if __name__ == "__main__":
    opt, config = parse_option()

    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.makedirs(opt.log_dir, exist_ok=True)

    os.environ["JOB_LOAD_DIR"] = os.path.dirname(config.load_path)

    logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name="pipework_eval")
    if dist.get_rank() == 0:
        path = os.path.join(config.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
            json.dump(vars(config), f, indent=2)
            os.system('cp %s %s' % (opt.cfg, config.log_dir))
        logger.info("Full config saved to {}".format(path))
    main(config, path)