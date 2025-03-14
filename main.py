import torch
import argparse
from torchstat import stat

from dtloader import OneDimDataset
from trainer import *
from utils import *

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Get available models from /model/network.py
model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch TFCNN For Bearing')
# dataset config
parser.add_argument('--data', default='../../dataset/CWRU', help='path to dataset.')
parser.add_argument('--data-mode', default='0HP 1HP 2HP 3HP', help='data split mode, which load'
                    ' or sense, contains "0HP 1HP 2HP 3HP" or "artificial", "reallife"')
parser.add_argument('--finetuneset', default="reallife", help='train dataset name')
parser.add_argument('--pretrainset', default='train', help='train dataset name')
parser.add_argument('--finetune-trainset', default='valid', help='val dataset name')
parser.add_argument('--finetune-testset', default='test', help='test dataset name')
parser.add_argument('--num-classes', default=10, type=int, help='num classes')
parser.add_argument('--length', default=2048, type=int, help='sample length.')
parser.add_argument('-o', '--overlap-rate', default=0.92, type=float, help='overlap rate.')
parser.add_argument('--train-samples', default=660, type=int, help='train samples.')
parser.add_argument('--test-samples', default=25, type=int, help='test samples.')
parser.add_argument('--valid-samples', default=10, type=int, help='valid samples.')
parser.add_argument('--snr', default=None, type=float, help='SNR.')
parser.add_argument('--do', default=None, type=float, help='Dropout rate.')
parser.add_argument('--dataaug', action='store_true', help='Activate data augmentation.')
parser.add_argument('--normalize-type', default="mean-std", type=str,
                    help='normalization for data')
parser.add_argument('--use-subset', action='store_true', help='Use PU sub dataset.')
parser.add_argument('--fft', action='store_true', help='Active frequency domain.')
parser.add_argument('--cwt', action='store_true', help='Active frequency domain.')
parser.add_argument('--use-saved-data', action='store_true', help='Active frequency domain.')

# backbone config
parser.add_argument('-a', '--backbone', metavar='ARCH', default='tfcnn',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: tfcnn)')
parser.add_argument('--mlp-hidden-size', default=128, type=int, metavar='N',
                    help='Mlp hidden size')
parser.add_argument('--projection-size', default=256, type=int, metavar='N',
                    help='Projection size')

# moco arch configs
parser.add_argument('--multimodal-loss', default='infonce-smoothl1', type=str, choices= \
                    ['infonce','smoothl1', 'infonce-smoothl1'], help='use which loss to optimize multimodal model.')
parser.add_argument('--l1-beta', default=0.01, type=float,
                    help='beta value for smoothL1 loss.')

# moco arch configs
parser.add_argument('--queue-size', default=8192, type=int, metavar='N',
                    help='queue size')
parser.add_argument('--queue-momentum', default=0.999, type=float, metavar='N',
                    help='moco momentum')

# clco configs
parser.add_argument('--ncentroids', default=20, type=int,
                    help='Kmeans centroid number')
parser.add_argument('--niter', default=100, type=int,
                    help='Kmeans iter number')
parser.add_argument('--select-data', default=0.5, type=float,
                    help='Kmeans iter number')
parser.add_argument('--loss', default='mpc', choices=['mpc', 'mcc', 'mse', 'bce'], 
                    type=str, help='multi label loss function')
parser.add_argument('--select-positive', default='combine_inst_dis', 
                    choices=['combine_inst_dis', 'only_inst_dis', 'cluster_postive'], 
                    type=str, help='add instance discrimination or only use instance '
                    'discrimination positive samples in physical hint method')
parser.add_argument('--random-pseudo-label', action='store_true', help='generate'
                    ' random pseudo label.')

# train options:
parser.add_argument('--train-mode', default='time-frequency', type=str, choices=[
                    'time-frequency', 'finetune', 'evaluate', 'analysis'], help='train mode.')
parser.add_argument('--resume-mode', default='time-frequency', type=str, choices=[
                    'time-frequency'], help='resume mode.')
parser.add_argument('--ae-mode', default='other', type=str, choices=['infonce_loss',
                    'infonce_loss_only', 'normal_contrast', 'other'], help='train mode.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--max-epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--finetune-batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer.')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--pretrain-lr-scheduler', action='store_true', 
                    help='active pretrain lr scheduler.')
parser.add_argument('--finetune-lr-scheduler', action='store_true', 
                    help='active pretrain lr scheduler.')
parser.add_argument('--base-lr', type=float, default=0.00001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum '
                    'for sgd')
parser.add_argument('--warm', default=5, type=int, help='lr scheduler warmup.')
parser.add_argument('--ftlr', default=0.001, type=float, help='finetune learning'
                    ' rate')
parser.add_argument('--ftwd', default=1e-6, type=float, 
                    help='finetune learning weight decay (default: 1e-6)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('-m', default=0.996, type=float,
                    help='EMA (default: 0.996)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--temp', default='temp', type=str, help='Checkpoint path.')
parser.add_argument('--active-log', action='store_true', help='active log.')
parser.add_argument('--patience', default=20, type=int, help='epoch number '
                    'detected for early stop.')

# finetune setting
parser.add_argument('--resume', default=None, help='resume checkpoint.')
parser.add_argument('--finetune-mode', default='train', help='resume checkpoint.')
parser.add_argument('--freeze', action='store_true', help='freeze fc parameter'
                    ' for random model initialization.')
parser.add_argument('--finetune-epochs', default=100, type=int, help='finetune'
                    ' epochs.')
parser.add_argument('--classifier', default=None, type=str, help='classifier type.')
parser.add_argument('--remark', default=None, help='remark.')

args = parser.parse_args()

args.device = f'cuda:{args.gpu_index}' if not args.disable_cuda and torch.cuda.is_available() else 'cpu'

print(f"Training with: {args.device}")

if args.active_log:
     args.writer = SummaryWriter(time_file(args.temp))

     # _create_model_training_folder(args.writer, files_to_same=['main.py', 'trainer.py', 'models'])
     args.checkpoint_dir = os.path.join(args.writer.log_dir, 'checkpoint.pt')
     logging.basicConfig(filename=os.path.join(args.writer.log_dir, 'training.log'), level=logging.DEBUG)
     for k, v in args.__dict__.items():
          logging.info("{}: {}".format(k, v)) 

model_param = [
          # 'layer5',
          'layer6'
          ]

def freeze_model(model, model_param, args):
     # freeze all layers but select model parameters
     if args.freeze:
          unfreeze_param = False
          for name, param in model.named_parameters():
               for mp in model_param:
                    if mp in name:
                         unfreeze_param = True
               if not unfreeze_param:
                      param.requires_grad = False
               unfreeze_param = False


if args.train_mode in ['time-frequency']:
     pretrain_dataset = OneDimDataset(args.pretrainset, args=args)
     time_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     args.in_channels = time_encoder.fc.in_features
     time_encoder.fc = models.mlphead(args)
     freq_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     freq_encoder.fc = models.mlphead(args)
     if args.backbone == 'tfcnn': args.in_channels = 2048
     time_projector = models.mlphead(args)
     freq_projector = models.mlphead(args)
     trainer = TimeFreqTrainer(args, time_encoder, freq_encoder, time_projector, freq_projector)
     trainer.train(pretrain_dataset)
     args.train_mode = "finetune"
     finetune_trainset = OneDimDataset(args.finetune_trainset, args=args)
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     time_projector = models.mlphead(args)
     freq_projector = models.mlphead(args)
     # time_projector = None
     # freq_projector = None
     args.in_channels = args.projection_size * 2
     args.projection_size = args.num_classes
     classifier = models.mlphead(args)
     args.resume = os.path.join(args.writer.log_dir, 'checkpoint_best.pt')
     if args.resume_mode == 'time-frequency':
          time_encoder, freq_encoder = load_timefreq(time_encoder, freq_encoder, args)
     else:
          model = load_sup(time_encoder, args)
     if time_projector and freq_projector:
          time_projector = load_sup(time_projector, args, 'time_projector')
          freq_projector = load_sup(freq_projector, args, 'freq_projector')
     freeze_model(time_encoder, model_param, args)
     freeze_model(freq_encoder, model_param, args)
     finetune_tffusion(time_encoder, freq_encoder, classifier, finetune_trainset, finetune_testset, args, time_projector, freq_projector)
elif args.train_mode == 'finetune':
     finetune_trainset = OneDimDataset(args.finetune_trainset, args=args)
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     if args.backbone == 'resnet18_1d': args.projection_size = 512
     time_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     freq_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     args.in_channels = time_encoder.fc.in_features
     time_encoder.fc = models.mlphead(args)
     freq_encoder.fc = models.mlphead(args)
     args.in_channels = args.projection_size
     if args.backbone == 'tfcnn': args.in_channels = 2048
     time_projector = models.mlphead(args)
     freq_projector = models.mlphead(args)
     # time_projector = None
     # freq_projector = None
     args.in_channels = args.projection_size * 2
     args.projection_size = args.num_classes
     classifier = models.mlphead(args)
     freeze_model(time_encoder, model_param, args)
     freeze_model(freq_encoder, model_param, args)
     if args.resume:
          if args.resume_mode == 'time-frequency':
               time_encoder, freq_encoder = load_timefreq(time_encoder, freq_encoder, args)
          else:
               model = load_sup(time_encoder, args)
          if time_projector and freq_projector:
               time_projector = load_sup(time_projector, args, 'time_projector')
               freq_projector = load_sup(freq_projector, args, 'freq_projector')
     finetune_tffusion(time_encoder, freq_encoder, classifier, finetune_trainset, finetune_testset, args, time_projector, freq_projector)
elif args.train_mode == 'evaluate':
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     val_loader = torch.utils.data.dataloader.DataLoader(finetune_testset,
                                                  batch_size=args.batch_size,
                                                  num_workers=0,
                                                  drop_last=False,
                                                  shuffle=False)
     time_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     args.in_channels = time_encoder.fc.in_features
     time_encoder.fc = models.mlphead(args)
     freq_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     freq_encoder.fc = models.mlphead(args)
     args.in_channels = args.projection_size * 2
     args.projection_size = args.num_classes
     classifier = models.mlphead(args)
     freeze_model(time_encoder, model_param, args)
     freeze_model(freq_encoder, model_param, args)
     if args.resume:
          if args.resume_mode == 'time-frequency':
               time_encoder, freq_encoder = load_timefreq(time_encoder, freq_encoder, args)
          else:
               model = load_sup(time_encoder, args)
     evaluate_tffusion(time_encoder, freq_encoder, classifier, val_loader, 200, args)
elif args.train_mode == 'analysis':
     finetune_trainset = OneDimDataset(args.finetune_trainset, args=args)
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     time_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     args.in_channels = time_encoder.fc.in_features
     time_encoder.fc = models.mlphead(args)
     freq_encoder = getattr(models, args.backbone)(num_classes=args.num_classes)
     freq_encoder.fc = models.mlphead(args)
     time_encoder = time_encoder.to(args.device).to(args.device)
     freq_encoder = freq_encoder.to(args.device).to(args.device)
     args.in_channels = args.projection_size * 2
     args.projection_size = args.num_classes
     classifier = models.mlphead(args).to(args.device)
     freeze_model(time_encoder, model_param, args)
     freeze_model(freq_encoder, model_param, args)
     if args.resume:
          if args.resume_mode == 'time-frequency':
               time_encoder, freq_encoder = load_timefreq(time_encoder, freq_encoder, args)
          else:
               model = load_sup(time_encoder, args)
     stat(model, input_size=(1, 2048))
     # analysis(model, finetune_trainset, finetune_testset, args)
else:
     raise NotImplementedError(f'There is no such self-supervised train strategy like {args.train_mode}!')

if args.active_log:
     args.writer.close()
