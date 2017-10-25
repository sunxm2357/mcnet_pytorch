import time
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import *
from util.visualizer import Visualizer

def main():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.print_freq = 1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# training videos = %d' % dataset_size)

    model =create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0  # total # of videos

    for epoch in range(opt.epoch_count, opt.nepoch + opt.nepoch_decay + 1):
        epoch_start_time = time.time()
        epoch_iters = 0  # # of videos in this epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time()-iter_start_time)/opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iters, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iters)/dataset_size, errors)

            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                video_name = data['video_name']
                visualizer.save_images(visuals, video_name, epoch, epoch_iters)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)


if __name__ == "__main__":
    main()




