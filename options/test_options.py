from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--batch_size", type=int, dest="batch_size", default=1, help="Mini-batch size")
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # Data Augment
        self.parser.add_argument('--data', required=True, type=str, help="name of test dataset [KTH|UCF]")
        self.parser.add_argument("--backwards", default=False, type=bool, help="play the video backwards")
        self.parser.add_argument("--pick_mode", default='Slide', type=str, help="pick up clip [Random|First|Slide]")
        self.parser.add_argument("--flip", default=False, type=bool, help="flip the frames in the videos")

        self.is_train = False