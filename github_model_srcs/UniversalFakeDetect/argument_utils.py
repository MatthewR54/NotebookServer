
def _base_args(parser):
    """
    Parse base arguments required for all training and evaluation.
    """
    # data sources
    parser.add_argument('--real_list_paths', default=None, action="extend", nargs="+", type=str, help='paths for lists of real images')
    parser.add_argument('--fake_list_paths', default=None, action="extend", nargs="+", type=str, help='paths for lists of fake images')
    
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--backbone', default='openai/clip-vit-large-patch14', type=str, help="huggingface id for backbone")
    parser.add_argument('--res', default=224, type=str, help="resolution for inference")
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--is_train', action='store_true', help='whether or not model is currently training.')
    parser.add_argument('--ckpt', type=str, default=None, help="path to stored checkpoint to load before evaluating or further training the model.")
    parser.add_argument('--backbone_crop', type=str, default=True, help="If the backbone should handle cropping or we should handle it on our end. The results are approximately equal.")
    parser.add_argument('--num_reals', type=int, default=None, help="number of reals to use, if specified.")
    parser.add_argument('--num_fakes', type=int, default=None, help="number of fakes to use, if specified.")
    parser.add_argument('--class_bal', action='store_true', help="defaults to false; when passed in, balances the class sizes. Mutually exclusive to num_reals and num_fakes.")
    parser.add_argument('--nworkers', type=int, default=4, help='number of workers (only used for training; num workers is auto set to 0 for evaluation)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='number of batches to load in-advance (only for training)')
    parser.add_argument('--data_aug', action='store_true', help='whether or not data augmentations should be applied.')
    return parser

def get_train_args(parser):
    """
    Additional arguments required for training.
    """
    parser = _base_args(parser)
    parser.add_argument('--loss_mult', type=float, default=1.0, help='multiplier for loss, e.g. if set to 2, then real (negative) loss is 2x fake (positive) loss')
    # parser.add_argument('--loss_freq', type=int, default=100, help='frequency of calculating loss')  # not used
    parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--niter', type=int, default=10, help='total epochs')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--log', action='store_true', help='log to wandb')
    
    # include additional real and fake paths when validating.
    parser.add_argument('--val_real_list_paths', default=None, action="extend", nargs="+", type=str, help='paths for lists of real images to use for validation')
    parser.add_argument('--val_fake_list_paths', default=None, action="extend", nargs="+", type=str, help='paths for lists of fake images to use for validation')
    
    # data augmentation
    parser.add_argument('--rz_interp', default='bilinear')
    parser.add_argument('--blur_prob', type=float, default=0.5)
    parser.add_argument('--blur_sig', default='0.0,3.0')
    parser.add_argument('--jpg_prob', type=float, default=0.5)
    parser.add_argument('--jpg_method', default='cv2,pil')
    parser.add_argument('--jpg_qual', default='30,100')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

    args = parser.parse_args()

    # additional processing
    args.rz_interp = args.rz_interp.split(',')
    args.blur_sig = [float(s) for s in args.blur_sig.split(',')]
    args.jpg_method = args.jpg_method.split(',')
    args.jpg_qual = [int(s) for s in args.jpg_qual.split(',')]
    if len(args.jpg_qual) == 2:
        args.jpg_qual = list(range(args.jpg_qual[0], args.jpg_qual[1] + 1))
    elif len(args.jpg_qual) > 2:
        raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

    return args

def get_eval_args(parser):
    """
    Evaluation specific arguments.
    """
    parser = _base_args(parser)
    parser.add_argument('--result_folder', type=str, default='result', help='where to save results to.')
    parser.add_argument('--find_thres', action="store_true", help="if set, finds best threshold and calculates results for that threshold in addition to the default of 0.5.")
    parser.add_argument('--save_files', action="store_true", help="if set, save copies of images to folders depending on classification.")
    args = parser.parse_args()
    args.is_train = False
    return args
