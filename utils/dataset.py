import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import open3d as o3d
import pickle

class ShapeNet(Dataset):
    def __init__(self, data_path, pc_path, n_points, subset, downsample=True):
        self.data_root = data_path
        self.pc_path = pc_path
        self.subset = subset
        self.npoints = 8192
        self.downsample = downsample

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = n_points
        self.whole = False
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })

        self.permutation = np.arange(self.sample_points_num)

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def uniform_down_sample(self, pts, down_sampling_nums):
        down_sampling_nums = self.npoints // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
        return np.asarray(ds_pcd.points)

    def geometric_down_sample(self, pts, down_sampling_nums):
        def vector_angle(x, y):
            Lx = np.sqrt(x.dot(x))
            Ly = (np.sum(y ** 2, axis=1)) ** 0.5
            cos_angle = np.sum(x * y, axis=1) / (Lx * Ly)
            angle = np.arccos(cos_angle)
            angle2 = angle * 360 / 2 / np.pi
            return angle2

        knn_num = 10
        angle_thre = 30
        down_sampling_ratio = self.npoints // down_sampling_nums

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        point_size = pts.shape[0]
        tree = o3d.geometry.KDTreeFlann(pcd)
        o3d.geometry.PointCloud.estimate_normals(
            pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_num))
        normal = np.asarray(pcd.normals)
        normal_angle = np.zeros((point_size))
        for i in range(point_size):
            [_, idx, dis] = tree.search_knn_vector_3d(pts[i], knn_num + 1)
            current_normal = normal[i]
            knn_normal = normal[idx[1:]]
            normal_angle[i] = np.mean(vector_angle(current_normal, knn_normal))

        high_condition = normal_angle >= angle_thre
        point_high = pts[np.where(high_condition)]
        point_low = pts[np.where(~high_condition)]

        pcd_high = o3d.geometry.PointCloud()
        pcd_high.points = o3d.utility.Vector3dVector(point_high)
        pcd_low = o3d.geometry.PointCloud()
        pcd_low.points = o3d.utility.Vector3dVector(point_low)
        pcd_high_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_high, down_sampling_ratio)
        pcd_low_down = o3d.geometry.PointCloud.uniform_down_sample(pcd_low, down_sampling_ratio)
        pcd_finl = o3d.geometry.PointCloud()
        pcd_finl.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_high_down.points),
                                                                     np.asarray(pcd_low_down.points))))
        return self.random_sample(np.asarray(pcd_finl.points), down_sampling_nums)


    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # data = self.random_sample(data, self.sample_points_num)
        if not self.downsample:
            data = self.random_sample(data, self.sample_points_num)
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'lr': data}
        elif not self.npoints == self.sample_points_num and self.downsample:
            lr = self.uniform_down_sample(data, self.sample_points_num)
            lr = pc_norm(lr)
            lr = torch.from_numpy(lr).float()
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'hr': data, 'lr': lr}
        else:
            data = pc_norm(data)
            data = torch.from_numpy(data).float()
            return {'hr': data}

    def __len__(self):
        return len(self.file_list)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def uniform_down_sample(pts, down_sampling_nums):
    down_sampling_nums = 16384 // down_sampling_nums
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
    return np.asarray(ds_pcd.points)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNet(Dataset):
    def __init__(self, root, number_pts, use_normal, cats, subset, downsampling=8192):
        self.root = root
        self.npoints = number_pts
        self.downsample = downsampling
        self.use_normals = use_normal
        self.num_category = cats
        self.process_data = True
        self.uniform = True
        split = subset
        self.subset = subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def uniform_down_sample(self, pts, down_sampling_nums):
        down_sampling_nums = 8192 // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_nums)
        return np.asarray(ds_pcd.points)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_norm(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        if self.downsample != self.npoints:
            point_set = self.uniform_down_sample(point_set, self.downsample)
        lr = self.uniform_down_sample(point_set, 2048)
        return {'model': point_set, 'label': label[0], 'lr': lr}


    def __getitem__(self, index):
        data = self._get_item(index)
        points = data['model']
        label = data['label']
        lr = data['lr']
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        lr = torch.from_numpy(lr).float()
        return {'model': current_points, 'label': label, 'lr': lr}


class Semantic_KITTI(Dataset):
    def __init__(self, datapath, subset, npoints=-1, npointslr=-1, ignore_index=-1, norm=False):
        self.npoints = npoints
        self.npointslr = npointslr
        self.subset = subset
        self.data_root = datapath
        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.norm = norm

    def _uniform_down_sample(self, pts, down_sampling_nums):

        down_sampling_factor = len(pts) // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_factor)

        clamp = np.asarray(ds_pcd.points)
        if len(clamp) <= down_sampling_nums:
            return clamp
        clamp_delete_num = len(clamp) - down_sampling_nums
        clamp_delete_elem = np.random.choice(len(clamp), clamp_delete_num, replace=False)
        clamp = np.delete(clamp, clamp_delete_elem, axis=0)

        return clamp

    """
    Ref: https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/semantic_kitti.py
    """
    def get_data_list(self):
        split2seq = dict(
            test=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            train=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.subset, str):
            seq_list = split2seq[self.subset]
        elif isinstance(self.subset, list):
            seq_list = []
            for split in self.subset:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        if self.norm:
            coord = pc_norm(coord)
        if self.npoints > -1 and self.npointslr > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
            coord_lr = self._uniform_down_sample(coord, self.npointslr)
            coord_tensor = torch.from_numpy(coord).float()
            coord_tensor_lr = torch.from_numpy(coord_lr).float()
            return dict(hr=coord_tensor, lr=coord_tensor_lr)

        if self.npoints > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
        coord_tensor = torch.from_numpy(coord).float()
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    def __getitem__(self, item):
        data_dict = self.get_data(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)


class KITTI_Object(Dataset):
    def __init__(self, datapath, subset, npoints=-1, npointslr=-1, ignore_index=-1, norm=False):
        self.npoints = npoints
        self.npointslr = npointslr
        self.subset = subset
        self.data_root = datapath
        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        # self.learning_map = self.get_learning_map(ignore_index)
        self.norm = norm

    def _uniform_down_sample(self, pts, down_sampling_nums):

        down_sampling_factor = len(pts) // down_sampling_nums
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        ds_pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, down_sampling_factor)

        clamp = np.asarray(ds_pcd.points)
        if len(clamp) <= down_sampling_nums:
            return clamp
        clamp_delete_num = len(clamp) - down_sampling_nums
        clamp_delete_elem = np.random.choice(len(clamp), clamp_delete_num, replace=False)
        clamp = np.delete(clamp, clamp_delete_elem, axis=0)

        return clamp

    """
    Ref: https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/semantic_kitti.py
    """
    def get_data_list(self):
        if self.subset == 'train':
            seq_folder = os.path.join(self.data_root, 'training')
        else:
            seq_folder = os.path.join(self.data_root, 'testing')
        data_list = []
        seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
        data_list += [
            os.path.join(seq_folder, "velodyne", file) for file in seq_files
        ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]

        if self.norm:
            coord = pc_norm(coord)
        if self.npoints > -1 and self.npointslr > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
            coord_lr = self._uniform_down_sample(coord, self.npointslr)
            coord_tensor = torch.from_numpy(coord).float()
            coord_tensor_lr = torch.from_numpy(coord_lr).float()
            return dict(hr=coord_tensor, lr=coord_tensor_lr)

        if self.npoints > -1:
            coord = self._uniform_down_sample(coord, self.npoints)
        coord_tensor = torch.from_numpy(coord).float()
        data_dict = dict(hr=coord_tensor)
        return data_dict

    def __getitem__(self, item):
        data_dict = self.get_data(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)


class ScanObjectNN(Dataset):
    """
    Ref: https://github.com/qizekun/ShapeLLM/blob/main/ReConV2/datasets/ScanObjectNNDataset.py
    """
    def __init__(self, data_path, subset):
        super().__init__()
        self.subset = subset
        self.root = data_path

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = pc_norm(current_points)
        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    train_dset = ShapeNet(
        data_path='../dataset/ShapeNet55/ShapeNet-55',
        pc_path='../dataset/ShapeNet55/shapenet_pc',
        subset='train',
        n_points=8192,
        downsample=True
    )
    val_dset = ShapeNet(
        data_path='../dataset/ShapeNet55/ShapeNet-55',
        pc_path='../dataset/ShapeNet55/shapenet_pc',
        subset='test',
        n_points=8192,
        downsample=True
    )
    val_loader = DataLoader(val_dset, batch_size=1, pin_memory=True)
    trn_loader = DataLoader(train_dset, batch_size=1, pin_memory=True)
    train_2k_set = []
    train_8k_set = []
    val_2k_set = []
    # val_8k_set = []
    # for i, batch in enumerate(tqdm(trn_loader, desc='Training set')):
    #     train_8k_set.append(batch['hr'].cpu().numpy())
    # for i, batch in enumerate(tqdm(val_loader, desc='Validate set')):
    #     val_8k_set.append(batch['hr'].cpu().numpy())

    # print(train_dset.__getitem__(0)['hr'].shape)
    test_KITTI = Semantic_KITTI(
        datapath='../dataset/SEMANTIC_KITTI_DIR',
        subset='train',
        norm=True,
        npoints=8192,
        npointslr=2048
    )
    # print(test_KITTI.__getitem__(0)['coord'])
    print(len(test_KITTI.__getitem__(0)['lr']))
    # testK_loader = DataLoader(test_KITTI, batch_size=1, pin_memory=True)

