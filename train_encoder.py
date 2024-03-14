import argparse
from datetime import datetime

from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.logger import *
from model.Encoder import *


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
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--log', type=bool, default=False)

# Grouping setting
parser.add_argument('--mask_type', type=str, default='rand')
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--group_size', type=int, default=32)
parser.add_argument('--num_group', type=int, default=64)
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--num_output', type=int, default=8192)

# Transformer setting
parser.add_argument('--trans_dim', type=int, default=384)
parser.add_argument('--drop_path_rate', type=float, default=0.1)

# Encoder setting
parser.add_argument('--encoder_depth', type=int, default=12)
parser.add_argument('--encoder_num_heads', type=int, default=6)
parser.add_argument('--encoder_dims', type=int, default=384)
parser.add_argument('--loss', type=str, default='cdl2')

# sche / optim
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--eta_min', type=float, default=0.000001)
parser.add_argument('--t_max', type=float, default=200)

args = parser.parse_args()
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
save_dir = os.path.join(args.save_dir, 'encocder_8192_width_{gSize}'.format(gSize=args.encoder_dims))

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

train_dset = ShapeNet(
    data_path='dataset/ShapeNet55/ShapeNet-55',
    pc_path='dataset/ShapeNet55/shapenet_pc',
    subset='train',
    n_points=2048,
    downsample=True
)
val_dset = ShapeNet(
    data_path='dataset/ShapeNet55/ShapeNet-55',
    pc_path='dataset/ShapeNet55/shapenet_pc',
    subset='test',
    n_points=2048,
    downsample=True
)

val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, pin_memory=True)
trn_loader = DataLoader(train_dset, batch_size=args.batch_size, pin_memory=True)
if args.log:
    log.info('Training decoder for stable diffusion.')
    log.info('config:')
    log.info(args)
    log.info('dataset loaded')
print('dataset loaded')


model = Encoder_Module(args).to(args.device)


optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.eta_min, T_max=args.t_max)


def train(i, batch, epoch):
    if i == 0 and epoch > 1:
        print('Saving checkpoint at epoch {epoch}'.format(epoch=epoch))
        save_model = {'args': args, 'model': model.state_dict()}
        if args.log:
            torch.save(save_model, os.path.join(save_dir, 'autoencoder_diff.pt'))

    x = batch['lr'].to(args.device)
    optimizer.zero_grad()
    model.train()
    loss = model(x, batch['hr'].to(args.device))
    loss.backward()
    optimizer.step()
    if args.log and i == 0:
        log.info('epoch: {epoch}, iteration: {i}, loss: {loss}'.format(i=i, epoch=epoch, loss=loss))
    print('epoch: {epoch}, iteration: {i}, loss: {loss}'.format(i=i, epoch=epoch, loss=loss))


def validate():
    all_recons = []
    for i, batch in enumerate(val_loader):
        print('sampling model {i}'.format(i=i))
        ref = batch['lr'].to(args.device)
        if i > 200:
            break
        with torch.no_grad():
            model.eval()
            x_vis, x_masked, mask, center, vis_pc, msk_pc = model.encode(ref, masked=False)
            recons = model.evaluate(x_vis, x_masked)
        all_recons.append(recons)
    all_recons = torch.cat(all_recons, dim=0)
    np.save(os.path.join(save_dir, 'out.npy'), all_recons.cpu().numpy())


try:
    n_it = 50
    epoch = 1
    while epoch <= n_it:
        model.train()
        for i, pc in enumerate(trn_loader):
            train(i, pc, epoch)
        scheduler.step()

        if epoch == n_it:
            if args.log:
                saved_file = {'args': args, 'model': model.state_dict()}
                torch.save(saved_file, os.path.join(save_dir, 'autoencoder_diff.pt'))
            validate()
        epoch += 1
except Exception as e:
    log.error(e)

