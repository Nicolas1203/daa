import argparse
import yaml

class Parser:
    """
    Command line parser based on argparse. This also includes arguments sanity check.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch implementation of Domain Aware Augmentations for Online Continual Learning")

        # Configuration parameters
        parser.add_argument('--config', default=None, help="Path to the configuration file for the training to launch.")
        # Training parameters
        parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help="Device to train on.")
        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--epochs', default=1, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=100, type=int,
                            help='stream batch size (default: 100)')
        parser.add_argument('--learning-rate', '-lr', default=0.1, type=float, help='Initial learning rate')
        parser.add_argument('--parallel', action='store_true', help="Whether to use every GPU available or not")
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--optim', default='SGD', choices=['Adam', 'SGD'])
        parser.add_argument('--test-freq', type=int, default=200, help="Nb batches between each save")
        parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
        parser.add_argument('--memory-only', '-mo', action='store_true', help='Training using only the memory ?')
        parser.add_argument('--supervised', action='store_true', help="Pseudo labels or true labels ?")
        # Logs parameters
        parser.add_argument('--tag', '-t', default='', help="Base name for graphs and checkpoints")
        parser.add_argument('--tb-root', default='./runs/', help="Where do you want tensorboards graphs ?")
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logs.')
        parser.add_argument('--logs-root', default='./logs', help="Defalt root folder for writing logs.")
        parser.add_argument('--results-root', default='./results/', help='Where you want to save the results ?')
        parser.add_argument('--tensorboard', action='store_true')
        # checkpoints params
        parser.add_argument('--save-freq', type=int, default=50, help='Frequency for saving models.')
        parser.add_argument('--ckpt-root', default='./checkpoints/', help='Directory where to save the model.')
        parser.add_argument('--resume', '-r', action='store_true', 
                            help="Resume old training. Setup model state and buffer state.")
        parser.add_argument('--model-state', help='Path/to/model/state.pth')
        parser.add_argument('--buffer-state', help='path/to/buffer/state.pth')
        # Test parameters
        parser.add_argument('--test', dest='train', action="store_false")
        ##########
        # MODELS #
        ##########
        parser.add_argument('--backbone', '-bb', default='rrn',
                            choices=['resnet18', 'rrn'],
                            help='Model backbone for contrastive training.')
        # Resnet parameters
        parser.add_argument('--nb-channels', type=int, default=3, 
                            help="Number of channels for the input image.")
        parser.add_argument('--no-proj', action='store_false', dest='proj', help="Do not use a projection layer for smiclr.")
        parser.add_argument('--proj', action='store_true', dest='proj', help="Use a projection layer for smiclr.")
        
        #####################
        # Dataset parameters
        #####################
        parser.add_argument('--dataset', '-d', default="cifar10", choices=['mnist', 'fmnist', 'cifar10', 'cifar100', 'tiny', 'sub'],
                            help='Dataset to train on')
        parser.add_argument('--data-root-dir', default='/data/',
                            help='Root dir containing the dataset to train on.')
        parser.add_argument('--min-crop', type=float, default=0.2, help="Minimum size for cropping in standard data augmentation. range (0-1)")
        parser.add_argument('--training-type', default='inc', choices=['uni', 'inc'],
                            help='How to feed the data to the network (incremental context or not)')
        parser.add_argument('--n-classes', type=int, default=10,
                            help="Number of classes in database.")
        parser.add_argument("--img-size", type=int, default=32, help="Size of the square input image")
        parser.add_argument('--num-workers', '-w', type=int, default=0, help='Number of workers to use for dataloader.')
        parser.add_argument('--data-norm', action='store_true', help="Normalize input data?")
        parser.add_argument("--n-tasks", type=int, default=5, help="How many tasks do you want?")
        parser.add_argument("--labels-order", type=int, nargs='+', help="In which order to you want to see the labels? Random if not specified.")
        # Contrastive loss parameters
        parser.add_argument('--temperature', default=0.07, type=float, 
                            metavar='T', help='temperature parameter for softmax')
        # Memory parameters
        parser.add_argument('--mem-size', type=int, default=200, help='Memory size for continual learning')  # used also for ltm
        parser.add_argument('--mem-batch-size', '-mbs', type=int, default=200, help="How many images do you want to retrieve from the memory/ltm")  # used also for ltm
        parser.add_argument('--buffer', default='reservoir', help="What buffer do you want? See available buffers in utils/name_match.py")
        parser.add_argument('--drop-method', default='random', choices=['random'], help="How to drop images from memory when adding new ones.")
        # Learner parameter
        parser.add_argument('--learner', help='What learner do you want? See list of available learners in utils/name_match.py')
        parser.add_argument('--eval-mem', action='store_true', dest='eval_mem')
        parser.add_argument('--eval-random', action='store_false', dest='eval_mem')
        # Multi runs arguments
        parser.add_argument('--n-runs', type=int, default=1, help="Number of runs, with different seeds each time.")
        parser.add_argument('--start-seed', type=int, default=0, help="First seed to use.")
        parser.add_argument('--run-id', type=int, help="Id of the current run in multi run.")
        parser.add_argument('--kornia', action='store_true', dest='kornia')
        parser.add_argument('--no-kornia', action='store_false', dest='kornia')
        # Inference parameters
        parser.add_argument('--lab-pc', type=int, default=20, help="Number of labeled images per class to use in unsupervised evaluation.")
        # Tricks
        parser.add_argument('--mem-iters', type=int, default=1, help="Number of times to make a grad update on memory at each step")
        parser.add_argument('--aug', action='store_true', dest='aug', help='Wanna have a batch with all images augmented or only half ?')
        parser.add_argument('--no-aug', action='store_false', dest="aug")
        # Multi aug params || DAA params
        parser.add_argument('--n-augs', type=int, default=2)
        parser.add_argument('--n-styles', type=int, default=0)
        parser.add_argument("--tf-size", type=int, default=128)
        parser.add_argument("--min-mix", type=float, default=0.5, help="Min proportion of the original image to keep when using mixup or cutmix.")
        parser.add_argument("--max-mix", type=float, default=1.0, help="Max proportion of the original image to keep when using mixup or cutmix.")
        parser.add_argument('--mixup', action='store_true')
        parser.add_argument('--n-mixup', type=int, default=0, help='Numbers of mixup to consider.')
        parser.add_argument('--cutmix', action='store_true')
        parser.add_argument('--n-cutmix', type=int, default=0, help='Numbers of cutmix to consider.')
        parser.add_argument('--multi-style', action='store_true')
        parser.add_argument("--style-samples", type=int, default=1, help="number of images to take the style from.")
        parser.add_argument("--min-style-alpha", type=float, default=1, help="Min alpha value for style transfer.")
        parser.add_argument("--max-style-alpha", type=float, default=1, help="Max alpha value for style transfer.")

        parser.set_defaults(train=True, proj=True, eval_mem=False, kornia=True, aug=True)
        self.parser = parser

    def parse(self, arguments=None):
        if arguments is not None:
            self.args = self.parser.parse_args(arguments)
        else:
            self.args = self.parser.parse_args()
        self.load_config()
        self.check_args()
        return self.args

    def load_config(self):
        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                cfg = yaml.safe_load(f)
                for key in cfg:
                    setattr(self.args, key, cfg[key])
            f.close()

    def check_args(self):
        """Modify default arguments values depending on the method and dataset.
        """
        #############################
        # Dataset parameters sanity #
        #############################
        if self.args.dataset == 'cifar10':
            self.args.img_size = 32
            self.args.n_classes = 10
        if self.args.dataset == 'cifar100': 
            self.args.img_size = 32
            self.args.n_classes = 100
        if self.args.dataset == 'tiny':
            self.args.img_size = 64
            self.args.n_classes = 200
            # Style parameters specific to tinyIN (TODO: add this to configs)
            self.min_style_alpha=0.4
            self.max_style_alpha=0.8
        if self.args.dataset == 'sub':
            self.args.img_size = 224
            self.args.n_classes = 10
            self.args.backbone = 'resnet18'
        ##############################
        # Learners parameters sanity #
        ##############################
        if self.args.learner == 'STAM':
            self.args.batch_size = 1

