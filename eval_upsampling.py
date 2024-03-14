import argparse
import os.path
from datetime import datetime

import point_cloud_utils as pcu
from glob import glob
from torch.utils.data import DataLoader
from torch import optim

from utils.dataset import *
from utils.logger import *
from model.DiffusionSR import *
from model.EncoderSR import *
from utils.visualization import *
from metrics.evaluation_metrics import compute_all_metrics, hausdorff_distance



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
parser.add_argument('--mask_ratio', type=float, default=0.4)
parser.add_argument('--group_size', type=int, default=32) # points in each group
parser.add_argument('--num_group', type=int, default=64) # number of group
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_output', type=int, default=8192)
parser.add_argument('--diffusion_output_size', default=8192)

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
    use_normal=False,
    cats=40,
    subset='test'
)
val_loader_MN = DataLoader(test_dset_MN, batch_size=args.val_batch_size, pin_memory=True)

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
check_point_dir = os.path.join('./pretrain_model/sr/encoder.pt')
check_point = torch.load(check_point_dir)['model']
encoder = Encoder_Module(args).to(args.device)
encoder.load_state_dict(check_point, strict=False)
print('model loaded')

diff_check_point_dir = os.path.join('./pretrain_model/sr/decoder.pt')
diff_check_point = torch.load(diff_check_point_dir)['model']
model = Diff_Point_MAE(args).to(args.device)
model = torch.nn.DataParallel(model, device_ids=[0])
incompatible = model.load_state_dict(diff_check_point, strict=False)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.000001, T_max=20)
PINK = [0xdd / 255, 0x83 / 255, 0xa2 / 255]
BLUE = [0x81 / 255, 0xb0 / 255, 0xbf / 255]
YELLOW = [0xf3 / 255, 0xdb / 255, 0x74 / 255]
CYAN = [0x2b / 255, 0xda / 255, 0xc0 / 255]


def load(filename, count=None):
    points = np.loadtxt(filename).astype(np.float32)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = uniform_down_sample(points, count)

    return points

def calculate_metric_all(size=-1):
    all_sample = []
    all_ref = []
    all_hd = []
    all_vis = []
    gt_paths = glob(os.path.join('dataset/PU1K/test/input_2048/gt_8192', '*.xyz'))
    x_paths = glob(os.path.join('dataset/PU1K/test/input_2048/input_2048', '*.xyz'))

    for i in range(0, len(gt_paths)):
        if not os.path.exists(save_dir + '/' + str(i)):
            os.makedirs(save_dir + '/' + str(i))
        with torch.no_grad():
            x_hr = torch.from_numpy(load(gt_paths[i])[:, :3]).float().unsqueeze(0).to('cuda')
            x = torch.from_numpy(load(x_paths[i])[:, :3]).float().unsqueeze(0).to('cuda')
            model.eval()
            x_vis, z_masked, mask, center, vis_pc, msk_pc = encoder.encode(x, masked=False)
            recons = model.module.sampling(x_vis, mask, center)
            all_vis.append(x_vis)
            all_sample.append(recons)
            all_ref.append(x_hr)
            hd = hausdorff_distance(recons, x_hr)
            all_hd.append(hd)
            print("evaluating model: {i}".format(i=i))

    mean_hd = sum(all_hd) / len(all_hd)
    sample = torch.cat(all_sample, dim=0)
    refpc = torch.cat(all_ref, dim=0)
    all = compute_all_metrics(sample, refpc, 1)
    print("MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'], N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))
    log.info("MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'], N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))

calculate_metric_all()
