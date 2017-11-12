import time
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import *
from util.visualizer import Visualizer
import pdb
from tensorboardX import SummaryWriter
from val import *


def main():
    # pdb.set_trace()
    opt, val_opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# training videos = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0  # total # of videos
    writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    for epoch in range(model.start_epoch, opt.nepoch + opt.nepoch_decay + 1):
        epoch_start_time = time.time()
        epoch_iters = 0  # # of videos in this epoch

        for i, data in enumerate(dataset):
            # pdb.set_trace()
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iters += opt.batch_size
            model.set_inputs(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time()-iter_start_time)/opt.batch_size
                writer.add_scalar('iter_time', t, total_steps / opt.batch_size)
                writer.add_scalars('loss', errors, total_steps/opt.batch_size)
                visualizer.print_current_errors(epoch, epoch_iters, errors, t)

            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                grid = visual_grid(visuals['seq_batch'], visuals['pred'], opt.K, opt.T)
                writer.add_image('current_batch', grid, total_steps / opt.batch_size)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest', epoch)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest', epoch)
            model.save(epoch, epoch)
            psnr_plot, ssim_plot, grid = val(val_opt)
            # pdb.set_trace()
            writer.add_image('psnr', psnr_plot, epoch)
            writer.add_image('ssim', ssim_plot, epoch)
            writer.add_image('samples', grid, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))


if __name__ == "__main__":
    main()




