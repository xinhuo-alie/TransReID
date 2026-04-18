import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class BallShow(BaseImageDataset):
    """
    自定义数据集加载类
    """
    # 这里对应 data/ 下的文件夹名称
    dataset_dir = 'BallShow'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(BallShow, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        # 处理文件夹
        # relabel=True 表示训练集会将 ID 重新映射为 0, 1, 2...
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print(f"=> BallShow loaded from {self.dataset_dir}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # 获取数据集统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)
        
        # 添加总摄像头数和总视角数
        self.num_total_cams = len(self.camid2label)
        self.num_total_vids = 1

    def _check_before_run(self):
        """检查文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths += glob.glob(osp.join(dir_path, '*.png'))  # 兼容png
        pattern = re.compile(r'([-\d]+)_c(\d+)')  # 支持多位摄像头 c10, c100

        # 第一次遍历：收集所有 pid 和 camid
        pid_container = set()
        camid_container = set()

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
            camid_container.add(camid)

        # 建立映射：让 camid 从 0 开始连续（关键！）
        cam2label = {cam: idx for idx, cam in enumerate(sorted(camid_container))}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        # 保存 cam2label 为类属性，以便后续使用
        if not hasattr(self, 'camid2label'):
            self.camid2label = cam2label

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue

            # 映射成连续的 camera id
            camid = cam2label[camid]

            if relabel:
                pid = pid2label[pid]
            else:
                pid = self.pid_begin + pid

            dataset.append((img_path, pid, camid, 1))

        return dataset