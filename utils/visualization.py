import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import Semantic_KITTI, KITTI_Object

PINK = [0xdd / 255, 0x83 / 255, 0xa2 / 255]
BLUE = [0x81 / 255, 0xb0 / 255, 0xbf / 255]
YELLOW = [0xf3 / 255, 0xdb / 255, 0x74 / 255]
CYAN = [0x2b / 255, 0xda / 255, 0xc0 / 255]

def plot_point_cloud(pc):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    return fig.show()


def show_model(pc, index=0, save_img=False, show_axis=True):
    """
    Load the generated result from npy file and show it in plots.
    Args:
        pc: the list of npy files which contain point cloud result. (Each npy file contains multiple PC model)
        for AE can take 2 files for output and reference; for GEN take 1 file for output results.
        index: show the specific results the group of point cloud result.
    """
    fig = plt.figure(figsize=(8, 8))
    for n_pc in range(0, len(pc)):
        ax = fig.add_subplot(1, len(pc), n_pc+1, projection='3d')
        models = np.load(pc[n_pc])
        print('Input file contains ' + str(len(models)) + ' models')
        print(len(models[0]))
        print(len(models))
        # print(len(models[1]))
        ax.scatter(models[index][:, 0], models[index][:, 1], models[index][:, 2])
        if n_pc == 0:
            ax.set_title('Output', fontsize=14)
        else:
            ax.set_title('Reference', fontsize=14)
        if not show_axis:
            ax.axis('off')
    if save_img:
        fig.savefig('output.png', dpi=300)
    fig.show()

def eval_model_normal(x, t, index=0, show_img=True, save_img=True, show_axis=True, save_location="", second_part=None, color=BLUE, color2=BLUE):
    """
    Load the generated result from npy file and show it in plots.
    Args:
        pc: the list of npy files which contain point cloud result. (Each npy file contains multiple PC model)
        for AE can take 2 files for output and reference; for GEN take 1 file for output results.
        index: show the specific results the group of point cloud result.
    """
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    models = x
    # ax.set_title('x_t', fontsize=14)
    ax.scatter(models[index][:, 0], models[index][:, 1], models[index][:, 2], color=color, marker='o', alpha=1.0)
    if second_part is not None:
        ax.scatter(second_part[index][:, 0], second_part[index][:, 1], second_part[index][:, 2], color=color2, marker='o', alpha=1.0)
    if not show_axis:
        ax.axis('off')
    if save_img:
        fig.savefig(save_location+'/{i}.png'.format(i=t), dpi=100)
    if show_img:
        fig.show()

def show_forward_diff(pc):
    fig = plt.figure(figsize=(20,8))
    print(len(pc))
    for i in range(0, len(pc)):
        if i % 10 == 0:
            ax = fig.add_subplot(1, 55, i+1, projection='3d')
            models = np.load(pc)
            ax.scatter(models[i][:, 0], models[i][:, 1], models[i][:, 2])
    fig.show()

if __name__ == '__main__':
    # test_KITTI = Semantic_KITTI(
    #     datapath='../dataset/SEMANTIC_KITTI_DIR',
    #     subset='test',
    #     norm=True,
    #     npoints=2048
    # )
    # print(test_KITTI.__len__())
    # plot_point_cloud(test_KITTI.__getitem__(4)['coord'])
    # show_model(['../results/pretrain_decoder_2024_02_28_17_43/out.npy'], 4)

    test_KITTI = KITTI_Object(
        datapath='../dataset/KITTI',
        subset='test',
        norm=True,

    )
    print(test_KITTI.__len__())
    plot_point_cloud(test_KITTI.__getitem__(90)['hr'])
    # show_model(['./results/new_start_2023_10_09_15_17/out.npy'], 21)
    # show_forward_diff('./results/new_stable_diffusion_2023_08_10_21_38/out.npy')

    # show_model(['./results/Maksed_noise_70/out.npy',
    #              './results/baseline/ref.npy'], 100)