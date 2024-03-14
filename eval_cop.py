from utils.dataset import *
from metrics.evaluation_metrics import averaged_hausdorff_distance, jsd_between_point_cloud_sets
from model.Completion import *
from model.EncoderCop import *
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from utils.logger import *

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


parser = argparse.ArgumentParser()
# Experiment setting
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--log', type=bool, default=True)

# Grouping setting
parser.add_argument('--mask_type', type=str, default='rand')
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--group_size', type=int, default=32) # points in each group
parser.add_argument('--num_group', type=int, default=64) # number of group
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_output', type=int, default=8192)

# Transformer setting
parser.add_argument('--trans_dim', type=int, default=384)
parser.add_argument('--drop_path_rate', type=float, default=0.1)

# Encoder setting
parser.add_argument('--encoder_depth', type=int, default=12)
parser.add_argument('--encoder_num_heads', type=int, default=6)
parser.add_argument('--loss', type=str, default='cdl2')

# Decoder setting
parser.add_argument('--decoder_depth', type=int, default=4)
parser.add_argument('--decoder_num_heads', type=int, default=4)

# diffusion
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')

args = parser.parse_args()
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_dir = os.path.join(args.save_dir, 'new_start_eval_{date}'.format(date=time_now))

if args.log:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s::%(levelname)s]  %(message)s')
    log_file = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    log_file.setLevel(logging.INFO)
    log_file.setFormatter(formatter)
    log.addHandler(log_file)

if args.log:
    log.info('loading dataset')
print('loading dataset')


test_dset_MN = ModelNet(
    root='dataset/ModelNet',
    number_pts=8192,
    downsampling=2048,
    use_normal=False,
    cats=40,
    subset='test'
)
val_loader_MN = DataLoader(test_dset_MN, batch_size=args.val_batch_size, pin_memory=True, shuffle=True)

val_dset = ShapeNet(
    data_path='dataset/ShapeNet55/ShapeNet-55',
    pc_path='dataset/ShapeNet55/shapenet_pc',
    subset='test',
    n_points=2048,
    downsample=True
)

val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, pin_memory=True)
if args.log:
    log.info('Training Stable Diffusion')
    log.info('config:')
    log.info(args)
    log.info('dataset loaded')
print('dataset loaded')

print('loading model')

def calculate_metric_all(size=-1, encoder=None):
    # mean_cd = []
    diff_check_point_dir = os.path.join('./pretrain_model/completion/decoder.pt')
    model = Diff_Point_MAE(args, encoder).to(args.device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.module.load_model_from_ckpt(diff_check_point_dir)


    all_sample = []
    all_ref = []
    all_hd = []
    all_cd = []
    for i, batch in enumerate(val_loader_MN):
        if i == size:
            break
        with torch.no_grad():
            ref = batch['model'].to(args.device)
            model.eval()
            x_vis, x_masked, mask, center, vis_pc, msk_pc = encoder.encode(ref)

            recons = model.module.sampling(x_vis)

            recons = torch.cat([vis_pc, recons], dim=1)
            all_sample.append(recons)
            all_ref.append(ref)
            hd = averaged_hausdorff_distance(recons.squeeze(), ref.squeeze())
            all_hd.append(hd)

            cd = chamfer_distance_l2(recons, ref)
            all_cd.append(cd)
            print("evaluating model: {i}, CD: {cd}".format(i=i, cd=cd))

    mean_hd = sum(all_hd) / len(all_hd)
    mean_cd = sum(all_cd) / len(all_cd)
    sample = torch.cat(all_sample, dim=0)
    refpc = torch.cat(all_ref, dim=0)
    jsd = jsd_between_point_cloud_sets(sample, refpc)
    print("MMD CD: {mmd}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=mean_cd, jsd=jsd, hd=mean_hd))
    log.info("MMD CD: {mmd}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=mean_cd, jsd=jsd, hd=mean_hd))


# calculate_metric_all()

def auto_run():
    lose_ratio = [0.75]
    for lr in lose_ratio:
        args.mask_ratio = lr
        check_point_dir = os.path.join('./pretrain_model/completion/encoder.pt')
        # check_point_dir = os.path.join(args.save_dir, '{saved_src}/encoder.pt'
        #                                .format(saved_src='new_start_eval_2023_11_12_02_21'))
        check_point = torch.load(check_point_dir)['model']
        encoder = Point_MAE(args).to(args.device)
        encoder.load_state_dict(check_point)
        print('model loaded')


        calculate_metric_all(encoder=encoder)

auto_run()
