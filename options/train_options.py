from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--lr", type=float, dest="lr", default=0.0001, help="Base Learning Rate")
        self.parser.add_argument("--alpha", type=float, dest="alpha", default=1.0, help="Image loss weight")
        self.parser.add_argument("--beta", type=float, dest="beta", default=0.02, help="GAN loss weight")
        self.parser.add_argument("--num_iter", type=int, dest="num_iter", default=100000, help="Number of iterations")
        self.parser.add_argument('--display_freq', type=int, default=1600, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=1600, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_policy', type=str, default= None, help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--margin', type=float, default=0.3, help="the margin used for choosing opt D or G")
        self.parser.add_argument('--nepoch', type=int, default=100, help='# of epoch at starting learning rate')
        self.parser.add_argument('--nepoch_decay', type=int, default=100, help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--model', type=str, default='mcnet', help='the model to run')
        self.parser.add_argument('--D_G_switch', type=str, default='adaptive', help='type of switching training in D and G [adaptive|alternative]')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--no_adversarial', action='store_true', help='do not use the adversarial loss')

        # Data Augment
        self.parser.add_argument('--train_data', required=True, type=str, help="name of training dataset [KTH]")
        self.parser.add_argument("--debug", default=False, type=bool, help="when debugging, overfit to the first training samples")
        self.parser.add_argument("--backwards", default=True, type=bool, help="play the video backwards")
        self.parser.add_argument("--pick_mode", default='Random', type=str, help="pick up clip [Random|First|Sequential]")
        self.parser.add_argument("--flip", default=True, type=bool, help="flip the frames in the videos")

        # TODO: add or delete
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')

        self.is_train = True