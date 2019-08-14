from FUNIT import FUNIT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of FUNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', choices=('train', 'test'), help='phase name')
    parser.add_argument('--dataset', type=str, default='flower', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=800000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch size for each gpu')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')
    parser.add_argument('--K', type=int, default=5, help='Test K')

    parser.add_argument('--gan_type', type=str, default='hinge', help='[gan / lsgan / hinge]')

    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--feature_weight', type=int, default=1, help='Weight about feature-matching')
    parser.add_argument('--recon_weight', type=int, default=0.1, help='Weight about reconstruction')

    parser.add_argument('--latent_dim', type=int, default=64, help='The dimension of class code')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')

    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--img_height', type=int, default=128, help='The height size of image')
    parser.add_argument('--img_width', type=int, default=128, help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = FUNIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")



if __name__ == '__main__':
    main()
