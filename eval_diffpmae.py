import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from utils.logger import *
from utils.dataset import *
from model.DiffusionPretrain import *
from model.Encoder import *
from utils.visualization import *
from metrics.evaluation_metrics import compute_all_metrics, averaged_hausdorff_distance, jsd_between_point_cloud_sets

parser = argparse.ArgumentParser()
# Experiment setting
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='./results')

# Grouping setting
parser.add_argument('--mask_type', type=str, default='rand')
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--group_size', type=int, default=32)  # points in each group
parser.add_argument('--num_group', type=int, default=64)  # number of group
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_output', type=int, default=8192)
parser.add_argument('--diffusion_output_size', type=int, default=2048)

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
check_point_dir = os.path.join('./pretrain_model/pretrain/encoder.pt')
check_point = torch.load(check_point_dir)['model']
encoder = Encoder_Module(args).to(args.device)
encoder.load_state_dict(check_point)
print('model loaded')

diff_check_point_dir = os.path.join('./pretrain_model/pretrain/decoder.pt')
diff_check_point = torch.load(diff_check_point_dir)['model']
model = Diff_Point_MAE(args).to(args.device)
model = torch.nn.DataParallel(model, device_ids=[0])
model.load_state_dict(diff_check_point)

PINK = [0xdd / 255, 0x83 / 255, 0xa2 / 255]
BLUE = [0x81 / 255, 0xb0 / 255, 0xbf / 255]
YELLOW = [0xf3 / 255, 0xdb / 255, 0x74 / 255]
CYAN = [0x2b / 255, 0xda / 255, 0xc0 / 255]
GRAY = [0xaa / 255, 0xaa / 255, 0xaa / 255]


def visiualizaion(index=-1):
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            if index == -1 or index == i:
                ref = batch['lr'].to(args.device)
                model.eval()
                encoder.eval()
                x_vis, z_masked, mask, center, vis_pc, msk_pc = encoder.encode(ref, masked=False)
                print('sampling model {i}'.format(i=i))
                t = torch.randint(0, args.num_steps, (ref.size(0),))
                recons = model.module.sampling(x_vis, mask, center, True)
                eval_model_normal(ref.cpu().numpy(), 'ori', show_axis=False, show_img=False, save_img=True,
                                  save_location=save_dir, color=BLUE)
                eval_model_normal(vis_pc.cpu().numpy(), 'msk', show_axis=False, show_img=False, save_img=True,
                                  save_location=save_dir, color=PINK)
                for i in range(0, len(recons)):
                    if i in [0, 10, 20, 30, 40, 50, 90, 150, 180]:
                        eval_model_normal(recons[i].cpu().numpy(), i, show_axis=False, show_img=True, save_img=True,
                                          save_location=save_dir, color=BLUE)
                # recons = torch.cat([vis_pc.reshape(1, -1, 3), recons], dim=1)
                eval_model_normal(recons[199].cpu().numpy(), 199, show_axis=False, show_img=False, save_img=True,
                                  save_location=save_dir, color=BLUE)
                # vis_results(recons, ref, vis_pc, i, show_axis=True, save_img=True, show_img=False)
                eval_model_normal(torch.cat([recons[199], vis_pc], dim=1).cpu().numpy(), -2, show_axis=False,
                                  show_img=False, save_img=True, save_location=save_dir, color=BLUE)
                eval_model_normal(recons[199].cpu().numpy(), 'comb', second_part=vis_pc.cpu().numpy(), show_axis=False,
                                  show_img=True, save_img=True, save_location=save_dir, color=BLUE, color2=PINK)
                eval_model_normal(msk_pc.cpu().numpy(), 'segment', second_part=vis_pc.cpu().numpy(), show_axis=False,
                                  show_img=False, save_img=True, save_location=save_dir, color=GRAY, color2=PINK)
        if index <= i:
            break

def calculate_metric_all_on_prediction(size=-1, enc=encoder, pred=model):
    all_sample = []
    all_ref = []
    all_hd = []
    for i, batch in enumerate(val_loader):
        if i == size:
            break
        with torch.no_grad():
            ref = batch['lr'].to(args.device)
            pred.eval()
            x_vis, z_masked, mask, center, vis_pc, msk_pc = enc.encode(ref, masked=False)
            recons = pred.module.sampling(x_vis, mask, center, False)
            recons = torch.cat([recons, vis_pc], dim=1)
            all_sample.append(recons)
            all_ref.append(ref)
            hd = averaged_hausdorff_distance(recons.squeeze(), ref.squeeze())
            all_hd.append(hd)

            print("evaluating model: {i}".format(i=i))
            log.info("evaluating model: {i}".format(i=i))

    mean_hd = sum(all_hd) / len(all_hd)
    sample = torch.cat(all_sample, dim=0)
    refpc = torch.cat(all_ref, dim=0)
    all = compute_all_metrics(sample, refpc, 32)
    print(
        "MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(
            mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'],
            N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))
    log.info(
        "MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(
            mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'],
            N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))


def auto_run():
    mask_ratio = [0.75]
    for v in mask_ratio:
        args.mask_ratio = v
        log.info("mask ratio: {mr}".format(mr=v))
        print('loading encoder')
        check_point_dir_l = os.path.join('./pretrain_model/pretrain/encoder.pt')
        check_point_l = torch.load(check_point_dir_l)['model']
        encoder_l = Encoder_Module(args).to(args.device)
        encoder_l.load_state_dict(check_point_l)
        print('encoder loaded\rloading model')

        diff_check_point_dir_l = os.path.join('./pretrain_model/pretrain/decoder.pt')
        diff_check_point_l = torch.load(diff_check_point_dir_l)['model']
        model_l = Diff_Point_MAE(args).to(args.device)
        model_l = torch.nn.DataParallel(model_l, device_ids=[0])
        model_l.load_state_dict(diff_check_point_l)
        print('model loaded')
        calculate_metric_all_on_prediction(2048, encoder_l, model_l)


# auto_run()

def calculate_metric_all_set():
    all_cd = []
    all_sample = []
    all_ref = []
    all_hd = []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            ref = batch['lr'].to(args.device)
            model.eval()
            encoder.eval()
            x_vis, z_masked, mask, center, vis_pc, msk_pc = encoder.encode(ref, masked=False)
            recons = model.module.sampling(x_vis, mask, center, False)
            recons = torch.cat([vis_pc, recons], dim=1)
            all_sample.append(recons)
            all_ref.append(ref)
            hd = averaged_hausdorff_distance(recons.squeeze(), ref.squeeze())
            all_hd.append(hd)

            cd = model.module.loss_func(recons, ref)
            all_cd.append(cd)

            print("evaluating model: {i}".format(i=i))
            # log.info("evaluating model: {i}, CD={cd}".format(i=i, cd=cd))

    mean_cd = sum(all_cd) / len(all_cd)
    mean_hd = sum(all_hd) / len(all_hd)
    sample = torch.cat(all_sample, dim=0)
    refpc = torch.cat(all_ref, dim=0)
    JSD = jsd_between_point_cloud_sets(sample, refpc)
    # all = compute_all_metrics(sample, refpc, 32)
    # print("MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'], N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))
    # log.info("MMD CD: {mmd}, \r\nCOV CD: {cov}, \r\nMMD-SMP CD: {mmd_smp}, \r\n1NN CD-t: {N_t}, \r\n1NN CD-f: {N_f}, \r\n1NN CD: {N}\r\nJSD: {jsd}\r\nHD: {hd}".format(mmd=all['lgan_mmd-CD'], cov=all['lgan_cov-CD'], mmd_smp=all['lgan_mmd_smp-CD'], N_t=all['1-NN-CD-acc_t'], N_f=all['1-NN-CD-acc_f'], N=all['1-NN-CD-acc'], jsd=all['JSD'], hd=mean_hd))
    print("Mean CD: {mCD}, Mean HD: {mHD}, JSD: {jsd}".format(mCD=mean_cd, mHD=mean_hd, jsd=JSD))
    log.info("Mean CD: {mCD}, Mean HD: {mHD}, JSD: {jsd}".format(mCD=mean_cd, mHD=mean_hd, jsd=JSD))

calculate_metric_all_set()