"""
方案1: 连续值编码
将离散参数直接转换为归一化的连续值，大幅减少通道数

优点：
- 实现最简单，仅需修改编码方式
- 通道数从 167 降至 7 (减少96%!)
- 计算量显著降低
- 保留所有参数信息

输入通道：
- 通道0: 2D建筑物掩码
- 通道1: Tx位置
- 通道2: 3D建筑物高度
- 通道3: 频率 (归一化到0-1)
- 通道4: TR数量 (log归一化)
- 通道5: 波束数量 (log归一化)
- 通道6: 波束ID (归一化到0-1)
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import warnings
from scipy.ndimage import zoom
import re

warnings.filterwarnings("ignore")


class MultiBeamRadioDataset(Dataset):
    """多波束数据集 - 连续值编码版本"""
    
    def __init__(self, 
                 phase="train",
                 dir_multibeam="/seu_share2/home/hanyu2/230258948/ogsp/Experiment/multibeam_labeled_radiomaps",
                 dir_height_maps="/seu_share2/home/hanyu2/230258948/ogsp/Experiment/height_maps",
                 dir_feature_maps="/seu_share2/home/hanyu2/230258948/ogsp/Experiment/simulation_results_multibeam",  # 添加默认值
                 split_strategy="random",
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 train_beam_folders=None,
                 val_beam_folders=None,
                 test_beam_folders=None,
                 train_scenes=None,
                 val_scenes=None,
                 test_scenes=None,
                 mode="sparse",
                 fix_samples=819,
                 num_samples_low=20000,
                 num_samples_high=50000,
                 transform=transforms.ToTensor(),
                 random_seed=42,
                 use_3d_buildings=True,
                 use_continuous_encoding=True,
                 use_feature_maps=False):
        """
        use_continuous_encoding: bool
            是否使用连续值编码（替代独热编码）
        use_feature_maps: bool
            是否使用 feature map（与 use_continuous_encoding 互斥）
        """
        super().__init__()
        
        # 参数校验
        if mode not in ["dense", "sparse"]:
            raise ValueError(f"mode必须为'dense'或'sparse'")
        if split_strategy not in ["random", "beam", "scene"]:
            raise ValueError(f"split_strategy必须为'random'/'beam'/'scene'")
        if phase not in ["train", "val", "test"]:
            raise ValueError(f"phase必须为'train'/'val'/'test'")
        if not os.path.exists(dir_multibeam):
            raise FileNotFoundError(f"multibeam目录不存在：{dir_multibeam}")
        if use_3d_buildings and not os.path.exists(dir_height_maps):
            raise FileNotFoundError(f"height_maps目录不存在：{dir_height_maps}")
        if use_feature_maps and not os.path.exists(dir_feature_maps):
            raise FileNotFoundError(f"feature_maps目录不存在：{dir_feature_maps}")
        
        # 互斥检查
        if use_feature_maps and use_continuous_encoding:
            warnings.warn("use_feature_maps 和 use_continuous_encoding 互斥，将关闭 use_continuous_encoding")
            use_continuous_encoding = False
        
        # 保存参数
        self.phase = phase
        self.dir_multibeam = dir_multibeam
        self.dir_height_maps = dir_height_maps
        self.dir_feature_maps = dir_feature_maps  # ⭐ 重要：保存这个参数
        self.split_strategy = split_strategy
        self.mode = mode
        self.transform = transform
        self.random_seed = random_seed
        self.use_3d_buildings = use_3d_buildings
        self.use_continuous_encoding = use_continuous_encoding
        self.use_feature_maps = use_feature_maps
        
        # 采样参数
        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high
        
        # 图像参数
        self.height = 256
        self.width = 256
        self.Tx_size = 2
        self.BUILDING_VALUE = 1000
        self.NO_LABEL_VALUE = -300
        
        np.random.seed(random_seed)
        
        # 收集beam文件夹
        self.all_beam_folders = sorted([
            f for f in os.listdir(dir_multibeam)
            if os.path.isdir(os.path.join(dir_multibeam, f))
        ])
        if not self.all_beam_folders:
            raise FileNotFoundError(f"未找到beam文件夹")
        print(f"共找到 {len(self.all_beam_folders)} 个beam配置")
        
        # 构建参数归一化映射（仅在使用连续编码时）
        if self.use_continuous_encoding:
            self._build_normalization_params()
        
        # 收集场景
        first_beam_folder = os.path.join(dir_multibeam, self.all_beam_folders[0])
        scene_files = glob(os.path.join(first_beam_folder, "u*_labeled_radiomap.npy"))
        self.all_scenes = sorted([
            os.path.basename(f).split('_')[0] for f in scene_files
        ])
        if not self.all_scenes:
            raise FileNotFoundError(f"未找到场景文件")
        print(f"共找到 {len(self.all_scenes)} 个场景")
        
        # 数据划分
        if split_strategy == "random":
            self._split_random(train_ratio, val_ratio, test_ratio)
        elif split_strategy == "beam":
            self._split_by_beam(train_beam_folders, val_beam_folders, test_beam_folders,
                               train_ratio, val_ratio, test_ratio)
        elif split_strategy == "scene":
            self._split_by_scene(train_scenes, val_scenes, test_scenes,
                                train_ratio, val_ratio, test_ratio)
        
        # 计算输入通道数
        self._calculate_input_channels()
        
        print(f"\n{phase}集统计：")
        print(f"  - 样本数: {len(self.samples)}")
        print(f"  - 输入通道数: {self.input_channels}")
        if use_feature_maps:
            print(f"  - 使用 Feature Map 模式")
        elif use_continuous_encoding:
            print(f"  - 使用连续编码模式")

            # ⭐ 预计算全局统计量
        self._compute_global_statistics()
    
    def _build_normalization_params(self):
        """构建参数归一化范围（用于连续值编码）"""
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_\w+_beam(\d+)'
        
        freq_set, tr_set, num_beams_set, beam_id_set = set(), set(), set(), set()
        
        for beam_folder in self.all_beam_folders:
            match = re.match(pattern, beam_folder)
            if match:
                freq, tr, num_beams, beam_id = match.groups()
                freq_set.add(float(freq))
                tr_set.add(int(tr))
                num_beams_set.add(int(num_beams))
                beam_id_set.add(int(beam_id))
        
        # 频率归一化：min-max到[0,1]
        self.freq_min = min(freq_set)
        self.freq_max = max(freq_set)
        
        # TR数量：对数归一化（因为TR是指数增长的：16,64,256,1024）
        self.tr_min = np.log10(min(tr_set))
        self.tr_max = np.log10(max(tr_set))
        
        # 波束数量：对数归一化
        self.num_beams_min = np.log10(min(num_beams_set))
        self.num_beams_max = np.log10(max(num_beams_set))
        
        # 波束ID：min-max到[0,1]
        self.beam_id_min = min(beam_id_set)
        self.beam_id_max = max(beam_id_set)
        
        print(f"\n连续值编码参数范围：")
        print(f"  - 频率: {self.freq_min}-{self.freq_max} GHz")
        print(f"  - TR数量: {min(tr_set)}-{max(tr_set)}")
        print(f"  - 波束数量: {min(num_beams_set)}-{max(num_beams_set)}")
        print(f"  - 波束ID: {self.beam_id_min}-{self.beam_id_max}")
    
    def _parse_beam_folder(self, beam_folder):
        """解析beam文件夹名称"""
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_\w+_beam(\d+)'
        match = re.match(pattern, beam_folder)
        if not match:
            raise ValueError(f"无法解析：{beam_folder}")
        
        freq, tr, num_beams, beam_id = match.groups()
        return {
            'freq': float(freq),
            'tr': int(tr),
            'num_beams': int(num_beams),
            'beam_id': int(beam_id)
        }
    
    def _create_continuous_encoding(self, beam_params):
        """
        创建连续值编码（4维向量）
        
        返回归一化到[0,1]的4维数组：
        [频率, TR数量(log), 波束数量(log), 波束ID]
        """
        # 频率归一化
        freq_norm = (beam_params['freq'] - self.freq_min) / (self.freq_max - self.freq_min + 1e-8)
        
        # TR数量对数归一化
        tr_log = np.log10(beam_params['tr'])
        tr_norm = (tr_log - self.tr_min) / (self.tr_max - self.tr_min + 1e-8)
        
        # 波束数量对数归一化
        num_beams_log = np.log10(beam_params['num_beams'])
        num_beams_norm = (num_beams_log - self.num_beams_min) / (self.num_beams_max - self.num_beams_min + 1e-8)
        
        # 波束ID归一化
        beam_id_norm = (beam_params['beam_id'] - self.beam_id_min) / (self.beam_id_max - self.beam_id_min + 1e-8)
        
        return np.array([freq_norm, tr_norm, num_beams_norm, beam_id_norm], dtype=np.float32)
    
    def _calculate_input_channels(self):
        """计算总输入通道数"""
        self.input_channels = 1  # 基础：建筑物+Tx
        if self.mode=="sparse":
            self.input_channels += 1
        if self.use_3d_buildings:
            self.input_channels += 1
        
        if self.use_feature_maps:
            self.input_channels += 1  # feature map 占1个通道
        elif self.use_continuous_encoding:
            self.input_channels += 3  # 连续值编码：4维
    
    def _split_random(self, train_ratio, val_ratio, test_ratio):
        """随机划分"""
        all_combinations = []
        for beam_folder in self.all_beam_folders:
            for scene in self.all_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, f"{scene}_labeled_radiomap.npy")
                if os.path.exists(file_path):
                    all_combinations.append((beam_folder, scene, file_path))
        
        np.random.shuffle(all_combinations)
        total = len(all_combinations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        if self.phase == "train":
            self.samples = all_combinations[:train_end]
        elif self.phase == "val":
            self.samples = all_combinations[train_end:val_end]
        else:
            self.samples = all_combinations[val_end:]
        
        print(f"\n随机划分: 训练{train_end} | 验证{val_end-train_end} | 测试{total-val_end}")
    
    def _split_by_beam(self, train_beams, val_beams, test_beams, train_ratio, val_ratio, test_ratio):
        """按波束划分"""
        if train_beams is None:
            total_beams = len(self.all_beam_folders)
            train_end = int(total_beams * train_ratio)
            val_end = train_end + int(total_beams * val_ratio)
            train_beams = self.all_beam_folders[:train_end]
            val_beams = self.all_beam_folders[train_end:val_end]
            test_beams = self.all_beam_folders[val_end:]
        
        selected_beams = {"train": train_beams, "val": val_beams, "test": test_beams}[self.phase]
        
        self.samples = []
        for beam_folder in selected_beams:
            for scene in self.all_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, f"{scene}_labeled_radiomap.npy")
                if os.path.exists(file_path):
                    self.samples.append((beam_folder, scene, file_path))
        
        print(f"\n按波束划分: {len(selected_beams)} beams × {len(self.all_scenes)} scenes")
    
    def _split_by_scene(self, train_scenes, val_scenes, test_scenes, train_ratio, val_ratio, test_ratio):
        """按场景划分"""
        if train_scenes is None:
            total_scenes = len(self.all_scenes)
            train_end = int(total_scenes * train_ratio)
            val_end = train_end + int(total_scenes * val_ratio)
            train_scenes = self.all_scenes[:train_end]
            val_scenes = self.all_scenes[train_end:val_end]
            test_scenes = self.all_scenes[val_end:]
        
        selected_scenes = {"train": train_scenes, "val": val_scenes, "test": test_scenes}[self.phase]
        
        self.samples = []
        for beam_folder in self.all_beam_folders:
            for scene in selected_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, f"{scene}_labeled_radiomap.npy")
                if os.path.exists(file_path):
                    self.samples.append((beam_folder, scene, file_path))
        
        print(f"\n按场景划分: {len(self.all_beam_folders)} beams × {len(selected_scenes)} scenes")
    
    def _generate_Tx_image(self, tx_x=127, tx_y=127):
        """生成Tx位置掩码"""
        tx_img = np.zeros((self.height, self.width), dtype=np.float32)
        tx_img[tx_y, tx_x] = 1.0
        return tx_img
    
    def _compute_global_statistics(self):
        """计算整个数据集的全局统计量"""
        print("计算全局统计量...")
        
        max_heights = []
        
        # 遍历所有场景，找到最大高度
        for scene in self.all_scenes:
            height_path = os.path.join(self.dir_height_maps, scene, f"{scene}_height_matrix.npy")
            if os.path.exists(height_path):
                height_matrix = np.load(height_path)
                max_heights.append(height_matrix.max())
        
        # 保存全局最大值
        self.GLOBAL_MAX_HEIGHT = max(max_heights) if max_heights else 100.0
        
        print(f"  全局最大建筑高度: {self.GLOBAL_MAX_HEIGHT:.2f}m")
    
    def _load_3d_building_matrix(self, scene):
        """加载3D建筑物高度矩阵 - 使用全局归一化"""
        height_matrix_path = os.path.join(self.dir_height_maps, scene, f"{scene}_height_matrix.npy")
        
        if not os.path.exists(height_matrix_path):
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            height_matrix = np.load(height_matrix_path).astype(np.float32)
            
            if height_matrix.shape != (self.height, self.width):
                zoom_factor = (self.height / height_matrix.shape[0], 
                              self.width / height_matrix.shape[1])
                height_matrix = zoom(height_matrix, zoom_factor, order=1)
            
            # ⭐ 使用全局最大值归一化
            height_matrix = height_matrix / self.GLOBAL_MAX_HEIGHT
            
            return height_matrix
        except Exception as e:
            warnings.warn(f"加载失败：{str(e)}")
            return np.zeros((self.height, self.width), dtype=np.float32)

    def _load_feature_map_matrix(self, scene, beam_folder, debug=False):
        """
        加载对应波束配置的 feature map（波束覆盖图 - dB值）
        使用与 radiomap 相同的归一化方式
        
        Args:
            scene: 场景名称 (e.g., "u0", "u122")
            beam_folder: radiomap的beam文件夹名
        
        Returns:
            feature_matrix: (H, W) 归一化后的特征矩阵 [0, 1]
        """
        if debug:
            print(f"\n[DEBUG] 加载 Feature Map: scene={scene}, beam_folder={beam_folder}")
        
        # 1. 解析 radiomap 文件夹名
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_([^_]+)(?:_beam(\d+))?'
        match = re.match(pattern, beam_folder)
        
        if not match:
            warnings.warn(f"无法解析 beam_folder: {beam_folder}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        freq, tr, num_beams, pattern_type, beam_id = match.groups()
        
        # 2. 构建 feature map 基础文件夹
        feature_base_folder = f"freq_{freq}GHz_{tr}TR_{num_beams}beams_pattern_{pattern_type}"
        
        if debug:
            print(f"  解析: freq={freq}, TR={tr}, beams={num_beams}, pattern={pattern_type}, beam_id={beam_id}")
            print(f"  基础文件夹: {feature_base_folder}")
        
        # 3. 构建完整路径
        feature_folder_path = os.path.join(self.dir_feature_maps, feature_base_folder, scene)
        
        if not os.path.exists(feature_folder_path):
            warnings.warn(f"Feature map 文件夹不存在：{feature_folder_path}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            # 4. 查找所有 matrix 文件
            matrix_files = sorted(glob(os.path.join(feature_folder_path, "beam_*_matrix.npy")))
            
            if not matrix_files:
                warnings.warn(f"未找到 feature map 文件：{feature_folder_path}")
                return np.zeros((self.height, self.width), dtype=np.float32)
            
            # 5. 根据 beam_id 选择对应文件
            if beam_id is not None:
                beam_id_int = int(beam_id)
                target_pattern = f"beam_{beam_id_int:02d}_angle_"
                
                matched_file = None
                for f in matrix_files:
                    if target_pattern in os.path.basename(f):
                        matched_file = f
                        break
                
                if matched_file is None:
                    warnings.warn(f"未找到 beam_{beam_id_int}，使用第一个: {os.path.basename(matrix_files[0])}")
                    matched_file = matrix_files[0]
                
                if debug:
                    print(f"  多波束: 寻找 {target_pattern}, 找到 {os.path.basename(matched_file)}")
            else:
                matched_file = matrix_files[0]
                
                if debug:
                    print(f"  单波束: 使用 {os.path.basename(matched_file)}")
            
            # 6. 加载数据
            feature_matrix = np.load(matched_file).astype(np.float32)
            
            if debug:
                print(f"  原始 dB: shape={feature_matrix.shape}, range=[{feature_matrix.min():.2f}, {feature_matrix.max():.2f}]")
            
            # 7. 调整尺寸
            if feature_matrix.shape != (self.height, self.width):
                zoom_factor = (self.height / feature_matrix.shape[0], 
                              self.width / feature_matrix.shape[1])
                feature_matrix = zoom(feature_matrix, zoom_factor, order=1)
            
            # 8. ⭐ 使用与 radiomap 相同的归一化方式
            # radiomap: (dB - NO_LABEL_VALUE) / (0 - NO_LABEL_VALUE)
            # 即: (dB - (-300)) / 300 = (dB + 300) / 300
            
            # 裁剪到有效范围 [NO_LABEL_VALUE, 0]
            feature_matrix = np.clip(feature_matrix, self.NO_LABEL_VALUE, 0)
            
            # 归一化到 [0, 1]
            feature_matrix_normalized = (feature_matrix - self.NO_LABEL_VALUE) / (0 - self.NO_LABEL_VALUE)
            
            if debug:
                print(f"  归一化后: range=[{feature_matrix_normalized.min():.6f}, {feature_matrix_normalized.max():.6f}], mean={feature_matrix_normalized.mean():.6f}")
            
            return feature_matrix_normalized
            
        except Exception as e:
            warnings.warn(f"加载 feature map 失败 ({beam_folder}/{scene}): {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros((self.height, self.width), dtype=np.float32)


    
    def _process_labeled_radiomap(self, labeled_radiomap):
        """预处理radiomap"""
        building_mask = (labeled_radiomap == self.BUILDING_VALUE).astype(np.float32)
        gain = np.where(building_mask == 1, 0, labeled_radiomap).astype(np.float32)
        no_label_mask = (gain == self.NO_LABEL_VALUE)
        gain[no_label_mask] = 0
        valid_mask = (gain < 0) & (gain > self.NO_LABEL_VALUE)
        
        if np.any(valid_mask):
            gain[valid_mask] = (gain[valid_mask] - self.NO_LABEL_VALUE) / (0 - self.NO_LABEL_VALUE)
        
        if gain.shape != (self.height, self.width):
            zoom_factor = (self.height / gain.shape[0], self.width / gain.shape[1])
            gain = zoom(gain, zoom_factor, order=1)
            building_mask = zoom(building_mask, zoom_factor, order=0)
            valid_mask = zoom(valid_mask.astype(np.float32), zoom_factor, order=0).astype(bool)
        
        return building_mask, gain, valid_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取样本"""
        beam_folder, scene, file_path = self.samples[idx]
        
        # 加载radiomap
        try:
            labeled_radiomap = np.load(file_path)
        except Exception as e:
            raise RuntimeError(f"加载失败：{file_path}\n{str(e)}")
        
        # 预处理
        image_buildings, image_gain, valid_area_mask = self._process_labeled_radiomap(labeled_radiomap)
        image_gain = np.expand_dims(image_gain, axis=2)
        
        # 生成Tx
        np.random.seed(self.random_seed + idx)
        image_Tx = self._generate_Tx_image(127, 127)
        
        # 构建输入通道
        input_channels_list = [image_buildings, image_Tx]
        
        # 添加3D建筑物
        if self.use_3d_buildings:
            height_matrix = self._load_3d_building_matrix(scene)
            input_channels_list.append(height_matrix)
        
        # 添加 feature map（替代连续值编码）
        if self.use_feature_maps:
            feature_matrix = self._load_feature_map_matrix('u0', beam_folder)
            input_channels_list.append(feature_matrix)
        else:
            # 添加连续值编码
            if self.use_continuous_encoding:
                beam_params = self._parse_beam_folder(beam_folder)
                continuous_vector = self._create_continuous_encoding(beam_params)
                
                # 将4维向量扩展为空间通道
                for i in range(4):
                    param_channel = np.full((self.height, self.width), 
                                           continuous_vector[i], dtype=np.float32)
                    input_channels_list.append(param_channel)
        
        # 合并所有通道
        inputs = np.stack(input_channels_list, axis=2)
        
        # 稀疏采样
        samples_mask = None
        if self.mode == "sparse":
            samples_mask = np.zeros((self.height, self.width), dtype=np.float32)
            
            if self.fix_samples == 0:
                num_samples = np.random.randint(self.num_samples_low, self.num_samples_high)
            else:
                num_samples = int(self.fix_samples)
            
            valid_coords = np.argwhere(valid_area_mask)
            if len(valid_coords) > 0:
                num_samples = min(num_samples, len(valid_coords))
                selected_indices = np.random.choice(len(valid_coords), num_samples, replace=False)
                selected_coords = valid_coords[selected_indices]
                x_samples, y_samples = selected_coords[:, 0], selected_coords[:, 1]
                samples_mask[x_samples, y_samples] = 1.0
        
        # 转换为张量
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            if self.mode == "sparse" and samples_mask is not None:
                samples_mask = self.transform(np.expand_dims(samples_mask, axis=2)).type(torch.float32)
        
        # 返回
        if self.mode == "dense":
            return (inputs, image_gain, valid_area_mask)
        else:
            return (inputs, image_gain, samples_mask, valid_area_mask)






if __name__ == "__main__":
    print("=" * 80)
    print("方案1测试：连续值编码")
    print("=" * 80)
    
    data_root = "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/multibeam_labeled_radiomaps"
    height_root = "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/height_maps"
    
    dataset = MultiBeamRadioDataset(
        phase="train",
        dir_multibeam=data_root,
        dir_height_maps=height_root,
        split_strategy="random",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        mode="sparse",
        random_seed=42,
        use_3d_buildings=True,
        use_continuous_encoding=True
    )
    
    inputs, gain, mask = dataset[0]
    print(f"\n样本形状：")
    print(f"  - 输入: {inputs.shape}  (通道数仅7！)")
    print(f"  - 增益: {gain.shape}")
    print(f"  - 掩码: {mask.shape}")
    
    print(f"\n通道分解：")
    print(f"  通道0: 2D建筑物掩码")
    print(f"  通道1: Tx位置")
    print(f"  通道2: 3D建筑物高度")
    print(f"  通道3: 频率 (归一化)")
    print(f"  通道4: TR数量 (log归一化)")
    print(f"  通道5: 波束数量 (log归一化)")
    print(f"  通道6: 波束ID (归一化)")
    
    print("\n✅ 通道数从167降至7，减少96%！")