"""
RME-GAN 数据集 - 改进版 v4
新增功能：
1. ✅ 支持 split_strategy: "random" / "beam" / "scene"
2. ✅ 支持 sparse 模式（稀疏采样）
3. ✅ Feature Map 支持两种波类型：球面波(spherical) / 平面波(plane)
4. ✅ 可配置采样点数量

关键特性：
- feature_matrix = self._load_feature_map_matrix('u0', beam_folder)  # 永远读u0
- split_strategy 与 MultiBeamRadioDataset 完全兼容
- wave_type: "spherical" / "plane" 选择不同的 Feature Map 来源

输入通道说明：
【use_feature_maps=True】
- Dense模式: [Tx, 高度, FeatureMap(u0)] → 3通道
- Sparse模式: [采样×GT, Tx, 高度, FeatureMap(u0)] → 4通道

【use_feature_maps=False】（选择维度1,2,3,4,6，去掉建筑物和num_beams）
- Dense模式: [Tx, 高度, freq, TR, beam_id] → 5通道
- Sparse模式: [采样×GT, Tx, 高度, freq, TR, beam_id] → 6通道
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


class RMEGANDataset(Dataset):
    """
    RME-GAN数据集 - 改进版 v4
    
    关键特性：
    1. Feature Map 永远读取 u0 场景（理论传播覆盖）
    2. 支持 dense/sparse 模式
    3. 支持 random/beam/scene 划分策略
    4. 全局归一化（建筑高度、feature map）
    5. 支持球面波/平面波 Feature Map 选择
    """
    
    def __init__(self, 
                 phase="train",
                 dir_multibeam="/path/to/multibeam_labeled_radiomaps",
                 dir_height_maps="/path/to/height_maps",
                 dir_feature_maps="/path/to/simulation_results_multibeam",
                 dir_feature_maps_plane="/path/to/Beam_Map_PlaneWave",  # ⭐ 新增：平面波路径
                 # ========== 新增参数 ==========
                 split_strategy="random",      # "random" / "beam" / "scene"
                 train_ratio=0.7,
                 val_ratio=0.1,
                 test_ratio=0.2,
                 train_beam_folders=None,      # beam划分时手动指定
                 val_beam_folders=None,
                 test_beam_folders=None,
                 train_scenes=None,            # scene划分时手动指定
                 val_scenes=None,
                 test_scenes=None,
                 # ========== sparse模式参数 ==========
                 mode="dense",                 # "dense" / "sparse"
                 fix_samples=819,              # 固定采样点数，0表示随机
                 num_samples_low=200,          # 随机采样下限
                 num_samples_high=1000,        # 随机采样上限
                 # ========== 原有参数 ==========
                 use_feature_maps=True,
                 wave_type="spherical",        # ⭐ 新增：波类型 "spherical" / "plane"
                 random_seed=42):
        """
        Args:
            phase: "train"/"val"/"test"
            split_strategy: "random"/"beam"/"scene" 划分策略
            mode: "dense"/"sparse" 采样模式
            fix_samples: sparse模式下的固定采样点数
            use_feature_maps: 是否使用feature map（强烈推荐True）
            wave_type: "spherical"(球面波) / "plane"(平面波) Feature Map来源
        """
        super().__init__()
        
        # 参数校验
        if phase not in ["train", "val", "test"]:
            raise ValueError(f"phase必须为'train'/'val'/'test'，当前: {phase}")
        if mode not in ["dense", "sparse"]:
            raise ValueError(f"mode必须为'dense'或'sparse'，当前: {mode}")
        if split_strategy not in ["random", "beam", "scene"]:
            raise ValueError(f"split_strategy必须为'random'/'beam'/'scene'，当前: {split_strategy}")
        if wave_type not in ["spherical", "plane"]:
            raise ValueError(f"wave_type必须为'spherical'或'plane'，当前: {wave_type}")
        
        self.phase = phase
        self.dir_multibeam = dir_multibeam
        self.dir_height_maps = dir_height_maps
        self.dir_feature_maps = dir_feature_maps  # 球面波路径
        self.dir_feature_maps_plane = dir_feature_maps_plane  # 平面波路径
        self.split_strategy = split_strategy
        self.mode = mode
        self.use_feature_maps = use_feature_maps
        self.wave_type = wave_type  # ⭐ 新增
        self.random_seed = random_seed
        
        # sparse模式参数
        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high
        
        # 图像参数
        self.height = 256
        self.width = 256
        self.BUILDING_VALUE = 1000
        self.NO_LABEL_VALUE = -300
        
        np.random.seed(random_seed)
        
        # 收集所有beam配置
        self.all_beam_folders = sorted([
            f for f in os.listdir(dir_multibeam)
            if os.path.isdir(os.path.join(dir_multibeam, f))
        ])
        if not self.all_beam_folders:
            raise FileNotFoundError(f"未找到beam文件夹: {dir_multibeam}")
        
        print(f"找到 {len(self.all_beam_folders)} 个beam配置")
        
        # 收集所有场景（从第一个beam文件夹）
        first_beam_folder = os.path.join(dir_multibeam, self.all_beam_folders[0])
        scene_files = glob(os.path.join(first_beam_folder, "u*_labeled_radiomap.npy"))
        self.all_scenes = sorted([
            os.path.basename(f).split('_')[0] for f in scene_files
        ])
        if not self.all_scenes:
            raise FileNotFoundError(f"未找到场景文件")
        
        print(f"找到 {len(self.all_scenes)} 个场景")
        
        # 根据策略划分数据集
        if split_strategy == "random":
            self._split_random(train_ratio, val_ratio, test_ratio)
        elif split_strategy == "beam":
            self._split_by_beam(train_beam_folders, val_beam_folders, test_beam_folders,
                               train_ratio, val_ratio, test_ratio)
        elif split_strategy == "scene":
            self._split_by_scene(train_scenes, val_scenes, test_scenes,
                                train_ratio, val_ratio, test_ratio)
        
        # 计算全局统计量（用于归一化）
        self._compute_global_statistics()
        
        # 输入通道数
        # 数据集返回: [建筑物, Tx, 高度, FeatureMap/编码参数]
        self.input_channels = 4 if use_feature_maps else 7  # 7 = 建筑物 + Tx + 高度 + 4个编码
        
        print(f"\n{phase}集配置:")
        print(f"  样本数: {len(self.samples)}")
        print(f"  输入通道数(数据集返回): {self.input_channels}")
        print(f"  划分策略: {split_strategy}")
        print(f"  采样模式: {mode}")
        if mode == "sparse":
            if fix_samples > 0:
                print(f"  固定采样点: {fix_samples}")
            else:
                print(f"  随机采样范围: [{num_samples_low}, {num_samples_high}]")
        if use_feature_maps:
            print(f"  ⭐ Feature Map 模式: 永远读取 u0 场景")
            print(f"  ⭐ 波类型: {wave_type} ({'球面波' if wave_type == 'spherical' else '平面波'})")
    
    # ========================================================================
    # 数据划分方法
    # ========================================================================
    
    def _split_random(self, train_ratio, val_ratio, test_ratio):
        """随机划分数据集"""
        all_combinations = []
        for beam_folder in self.all_beam_folders:
            for scene in self.all_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, 
                                        f"{scene}_labeled_radiomap.npy")
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
    
    def _split_by_beam(self, train_beams, val_beams, test_beams,
                       train_ratio, val_ratio, test_ratio):
        """按波束配置划分（测试泛化到新波束）"""
        if train_beams is None:
            # 自动划分beam
            total_beams = len(self.all_beam_folders)
            beam_folders_shuffled = self.all_beam_folders.copy()
            np.random.shuffle(beam_folders_shuffled)
            
            train_end = int(total_beams * train_ratio)
            val_end = train_end + int(total_beams * val_ratio)
            
            train_beams = beam_folders_shuffled[:train_end]
            val_beams = beam_folders_shuffled[train_end:val_end]
            test_beams = beam_folders_shuffled[val_end:]
            
            print(f"\n自动按Beam划分: 训练{len(train_beams)} | 验证{len(val_beams)} | 测试{len(test_beams)}")
        else:
            print(f"\n手动按Beam划分: 训练{len(train_beams)} | 验证{len(val_beams)} | 测试{len(test_beams)}")
        
        selected_beams = {"train": train_beams, "val": val_beams, "test": test_beams}[self.phase]
        
        self.samples = []
        for beam_folder in selected_beams:
            for scene in self.all_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, 
                                        f"{scene}_labeled_radiomap.npy")
                if os.path.exists(file_path):
                    self.samples.append((beam_folder, scene, file_path))
        
        print(f"  {self.phase}集: {len(selected_beams)} beams × {len(self.all_scenes)} scenes = {len(self.samples)} 样本")
    
    def _split_by_scene(self, train_scenes, val_scenes, test_scenes,
                        train_ratio, val_ratio, test_ratio):
        """按场景划分（测试泛化到新场景）"""
        if train_scenes is None:
            # 自动划分scene
            all_scenes_copy = self.all_scenes.copy()
            np.random.shuffle(all_scenes_copy)
            
            total_scenes = len(all_scenes_copy)
            train_end = int(total_scenes * train_ratio)
            val_end = train_end + int(total_scenes * val_ratio)
            
            train_scenes = all_scenes_copy[:train_end]
            val_scenes = all_scenes_copy[train_end:val_end]
            test_scenes = all_scenes_copy[val_end:]
            
            print(f"\n自动按Scene划分: 训练{len(train_scenes)} | 验证{len(val_scenes)} | 测试{len(test_scenes)}")
        else:
            print(f"\n手动按Scene划分: 训练{len(train_scenes)} | 验证{len(val_scenes)} | 测试{len(test_scenes)}")
        
        selected_scenes = {"train": train_scenes, "val": val_scenes, "test": test_scenes}[self.phase]
        
        self.samples = []
        for beam_folder in self.all_beam_folders:
            for scene in selected_scenes:
                file_path = os.path.join(self.dir_multibeam, beam_folder, 
                                        f"{scene}_labeled_radiomap.npy")
                if os.path.exists(file_path):
                    self.samples.append((beam_folder, scene, file_path))
        
        print(f"  {self.phase}集: {len(self.all_beam_folders)} beams × {len(selected_scenes)} scenes = {len(self.samples)} 样本")
    
    # ========================================================================
    # 统计和加载方法
    # ========================================================================
    
    def _compute_global_statistics(self):
        """计算全局最大建筑高度（用于归一化）"""
        print("计算全局统计量...")
        max_heights = []
        
        for scene in self.all_scenes:
            height_path = os.path.join(self.dir_height_maps, scene, 
                                      f"{scene}_height_matrix.npy")
            if os.path.exists(height_path):
                height_matrix = np.load(height_path)
                max_heights.append(height_matrix.max())
        
        self.GLOBAL_MAX_HEIGHT = max(max_heights) if max_heights else 100.0
        print(f"  全局最大建筑高度: {self.GLOBAL_MAX_HEIGHT:.2f}m")
    
    def _load_3d_building_matrix(self, scene):
        """加载3D建筑物高度矩阵（全局归一化）"""
        height_matrix_path = os.path.join(self.dir_height_maps, scene, 
                                         f"{scene}_height_matrix.npy")
        
        if not os.path.exists(height_matrix_path):
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            height_matrix = np.load(height_matrix_path).astype(np.float32)
            
            if height_matrix.shape != (self.height, self.width):
                zoom_factor = (self.height / height_matrix.shape[0], 
                              self.width / height_matrix.shape[1])
                height_matrix = zoom(height_matrix, zoom_factor, order=1)
            
            # 全局归一化
            height_matrix = height_matrix / self.GLOBAL_MAX_HEIGHT
            
            return height_matrix
        except Exception as e:
            warnings.warn(f"加载高度矩阵失败: {str(e)}")
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def _load_feature_map_matrix_spherical(self, scene, beam_folder):
        """
        ⭐ 加载球面波 Feature Map（原有方式）
        
        Args:
            scene: 应该永远是 'u0'
            beam_folder: radiomap的beam文件夹名
        
        Returns:
            feature_matrix: (H, W) 归一化后的特征矩阵 [0, 1]
        """
        # 解析 radiomap 文件夹名
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_([^_]+)(?:_beam(\d+))?'
        match = re.match(pattern, beam_folder)
        
        if not match:
            warnings.warn(f"无法解析 beam_folder: {beam_folder}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        freq, tr, num_beams, pattern_type, beam_id = match.groups()
        
        # 构建 feature map 基础文件夹
        feature_base_folder = f"freq_{freq}GHz_{tr}TR_{num_beams}beams_pattern_{pattern_type}"
        
        # 构建完整路径
        feature_folder_path = os.path.join(self.dir_feature_maps, 
                                          feature_base_folder, scene)
        
        if not os.path.exists(feature_folder_path):
            warnings.warn(f"球面波 Feature map 文件夹不存在: {feature_folder_path}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            # 查找所有 matrix 文件
            matrix_files = sorted(glob(os.path.join(feature_folder_path, 
                                                    "beam_*_matrix.npy")))
            
            if not matrix_files:
                warnings.warn(f"未找到球面波 feature map 文件: {feature_folder_path}")
                return np.zeros((self.height, self.width), dtype=np.float32)
            
            # 根据 beam_id 选择对应文件
            if beam_id is not None:
                beam_id_int = int(beam_id)
                target_pattern = f"beam_{beam_id_int:02d}_angle_"
                
                matched_file = None
                for f in matrix_files:
                    if target_pattern in os.path.basename(f):
                        matched_file = f
                        break
                
                if matched_file is None:
                    matched_file = matrix_files[0]
            else:
                matched_file = matrix_files[0]
            
            # 加载数据
            feature_matrix = np.load(matched_file).astype(np.float32)
            
            # 调整尺寸
            if feature_matrix.shape != (self.height, self.width):
                zoom_factor = (self.height / feature_matrix.shape[0], 
                              self.width / feature_matrix.shape[1])
                feature_matrix = zoom(feature_matrix, zoom_factor, order=1)
            
            # 使用与 radiomap 相同的归一化方式
            feature_matrix = np.clip(feature_matrix, self.NO_LABEL_VALUE, 0)
            feature_matrix_normalized = (feature_matrix - self.NO_LABEL_VALUE) / \
                                       (0 - self.NO_LABEL_VALUE)
            
            return feature_matrix_normalized
            
        except Exception as e:
            warnings.warn(f"加载球面波 feature map 失败: {str(e)}")
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def _load_feature_map_matrix_plane(self, scene, beam_folder):
        """
        ⭐⭐⭐ 新增：加载平面波 Feature Map
        
        平面波路径格式: /Beam_Map_PlaneWave/{beam_folder}/beam_{id}_matrix.npy
        
        Args:
            scene: 'u0'（平面波不区分场景，因为是理论覆盖）
            beam_folder: radiomap的beam文件夹名
        
        Returns:
            feature_matrix: (H, W) 归一化后的特征矩阵 [0, 1]
        """
        # 解析 radiomap 文件夹名获取 beam_id
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_([^_]+)(?:_beam(\d+))?'
        match = re.match(pattern, beam_folder)
        
        if not match:
            warnings.warn(f"无法解析 beam_folder: {beam_folder}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        freq, tr, num_beams, pattern_type, beam_id = match.groups()
        
        # 构建平面波 feature map 路径
        # 格式: /Beam_Map_PlaneWave/{beam配置文件夹}/beam_{id}_matrix.npy
        feature_base_folder = f"freq_{freq}GHz_{tr}TR_{num_beams}beams_pattern_{pattern_type}"
        feature_folder_path = os.path.join(self.dir_feature_maps_plane, feature_base_folder)
        
        if not os.path.exists(feature_folder_path):
            warnings.warn(f"平面波 Feature map 文件夹不存在: {feature_folder_path}")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        try:
            # 查找所有 matrix 文件
            matrix_files = sorted(glob(os.path.join(feature_folder_path, "beam_*_matrix.npy")))
            
            if not matrix_files:
                warnings.warn(f"未找到平面波 feature map 文件: {feature_folder_path}")
                return np.zeros((self.height, self.width), dtype=np.float32)
            
            # 根据 beam_id 选择对应文件
            if beam_id is not None:
                beam_id_int = int(beam_id)
                # 尝试多种命名格式
                possible_patterns = [
                    f"beam_{beam_id_int:02d}_",
                    f"beam_{beam_id_int}_",
                    f"beam{beam_id_int:02d}_",
                    f"beam{beam_id_int}_"
                ]
                
                matched_file = None
                for f in matrix_files:
                    filename = os.path.basename(f)
                    for pat in possible_patterns:
                        if pat in filename:
                            matched_file = f
                            break
                    if matched_file:
                        break
                
                if matched_file is None:
                    # 按索引选择
                    if beam_id_int < len(matrix_files):
                        matched_file = matrix_files[beam_id_int]
                    else:
                        matched_file = matrix_files[0]
            else:
                matched_file = matrix_files[0]
            
            # 加载数据
            feature_matrix = np.load(matched_file).astype(np.float32)
            
            # 调整尺寸
            if feature_matrix.shape != (self.height, self.width):
                zoom_factor = (self.height / feature_matrix.shape[0], 
                              self.width / feature_matrix.shape[1])
                feature_matrix = zoom(feature_matrix, zoom_factor, order=1)
            
            # 归一化：根据数据范围自适应
            data_min = feature_matrix.min()
            data_max = feature_matrix.max()
            
            if data_max > data_min:
                # 如果数据在dB范围（如 -300 到 0）
                if data_min < -100:
                    feature_matrix = np.clip(feature_matrix, self.NO_LABEL_VALUE, 0)
                    feature_matrix_normalized = (feature_matrix - self.NO_LABEL_VALUE) / \
                                               (0 - self.NO_LABEL_VALUE)
                else:
                    # 通用归一化
                    feature_matrix_normalized = (feature_matrix - data_min) / (data_max - data_min)
            else:
                feature_matrix_normalized = np.zeros_like(feature_matrix)
            
            return feature_matrix_normalized
            
        except Exception as e:
            warnings.warn(f"加载平面波 feature map 失败: {str(e)}")
            return np.zeros((self.height, self.width), dtype=np.float32)
    
    def _load_feature_map_matrix(self, scene, beam_folder):
        """
        ⭐⭐⭐ 统一接口：根据 wave_type 加载对应的 Feature Map
        
        Args:
            scene: 应该永远是 'u0'
            beam_folder: radiomap的beam文件夹名
        
        Returns:
            feature_matrix: (H, W) 归一化后的特征矩阵 [0, 1]
        """
        if self.wave_type == "spherical":
            return self._load_feature_map_matrix_spherical(scene, beam_folder)
        elif self.wave_type == "plane":
            return self._load_feature_map_matrix_plane(scene, beam_folder)
        else:
            raise ValueError(f"未知的波类型: {self.wave_type}")
    
    def _process_labeled_radiomap(self, labeled_radiomap):
        """预处理radiomap"""
        # 提取建筑物掩码
        building_mask = (labeled_radiomap == self.BUILDING_VALUE).astype(np.float32)
        
        # 提取增益值
        gain = np.where(building_mask == 1, 0, labeled_radiomap).astype(np.float32)
        no_label_mask = (gain == self.NO_LABEL_VALUE)
        gain[no_label_mask] = 0
        
        # 有效区域掩码
        valid_mask = (gain < 0) & (gain > self.NO_LABEL_VALUE)
        
        # 归一化增益值到[0,1]
        if np.any(valid_mask):
            gain[valid_mask] = (gain[valid_mask] - self.NO_LABEL_VALUE) / \
                              (0 - self.NO_LABEL_VALUE)
        
        # 调整尺寸
        if gain.shape != (self.height, self.width):
            zoom_factor = (self.height / gain.shape[0], 
                          self.width / gain.shape[1])
            gain = zoom(gain, zoom_factor, order=1)
            building_mask = zoom(building_mask, zoom_factor, order=0)
            valid_mask = zoom(valid_mask.astype(np.float32), 
                            zoom_factor, order=0).astype(bool)
        
        return building_mask, gain, valid_mask
    
    def _generate_Tx_image(self, tx_x=127, tx_y=127):
        """生成Tx位置掩码"""
        tx_img = np.zeros((self.height, self.width), dtype=np.float32)
        tx_img[tx_y, tx_x] = 1.0
        return tx_img
    
    def _parse_beam_folder(self, beam_folder):
        """解析beam文件夹名称"""
        pattern = r'freq_([\d.]+)GHz_(\d+)TR_(\d+)beams_pattern_\w+_beam(\d+)'
        match = re.match(pattern, beam_folder)
        if not match:
            raise ValueError(f"无法解析: {beam_folder}")
        
        freq, tr, num_beams, beam_id = match.groups()
        return {
            'freq': float(freq),
            'tr': int(tr),
            'num_beams': int(num_beams),
            'beam_id': int(beam_id)
        }
    
    def _create_continuous_encoding(self, beam_params):
        """创建连续值编码（4维向量）"""
        # 简单的min-max归一化
        freq_norm = (beam_params['freq'] - 2.0) / (30.0 - 2.0)
        tr_norm = np.log10(beam_params['tr'] / 16) / np.log10(1024 / 16)
        num_beams_norm = (beam_params['num_beams'] - 1) / 64
        beam_id_norm = beam_params['beam_id'] / 64
        
        return np.array([freq_norm, tr_norm, num_beams_norm, beam_id_norm], 
                       dtype=np.float32)
    
    def _generate_sparse_samples(self, valid_area_mask, idx):
        """
        生成稀疏采样掩码
        
        Args:
            valid_area_mask: 有效区域掩码 (H, W)
            idx: 样本索引（用于确定性随机）
        
        Returns:
            samples_mask: 采样点掩码 (H, W)
        """
        # 使用确定性随机种子
        np.random.seed(self.random_seed + idx)
        
        samples_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 获取有效坐标
        valid_coords = np.argwhere(valid_area_mask)
        
        if len(valid_coords) == 0:
            return samples_mask
        
        # 确定采样数量
        if self.fix_samples > 0:
            num_samples = min(self.fix_samples, len(valid_coords))
        else:
            num_samples = np.random.randint(self.num_samples_low, self.num_samples_high + 1)
            num_samples = min(num_samples, len(valid_coords))
        
        # 随机选择采样点
        selected_indices = np.random.choice(len(valid_coords), num_samples, replace=False)
        selected_coords = valid_coords[selected_indices]
        
        # 设置采样掩码
        samples_mask[selected_coords[:, 0], selected_coords[:, 1]] = 1.0
        
        return samples_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取样本
        
        返回:
            dense模式: (inputs, targets, valid_mask)
            sparse模式: (inputs, targets, samples_mask, valid_mask)
            
        注意：inputs的维度是 [C, H, W]，其中:
            - 通道0: 建筑物掩码
            - 通道1: Tx位置
            - 通道2: 3D建筑高度
            - 通道3: Feature Map (u0场景) 或 频率参数
            - 通道4-6: (仅无feature map时) TR/波束数/波束ID
        """
        beam_folder, scene, file_path = self.samples[idx]
        
        # 加载radiomap
        try:
            labeled_radiomap = np.load(file_path)
        except Exception as e:
            raise RuntimeError(f"加载失败: {file_path}\n{str(e)}")
        
        # 预处理
        image_buildings, image_gain, valid_area_mask = \
            self._process_labeled_radiomap(labeled_radiomap)
        
        # 生成Tx
        np.random.seed(self.random_seed + idx)
        image_Tx = self._generate_Tx_image(127, 127)
        
        # 构建输入通道
        inputs = [image_buildings, image_Tx]
        
        # 添加3D建筑物高度
        height_matrix = self._load_3d_building_matrix(scene)
        inputs.append(height_matrix)
        
        # ⭐⭐⭐ 添加 feature map（永远读u0！）
        if self.use_feature_maps:
            feature_matrix = self._load_feature_map_matrix('u0', beam_folder)
            inputs.append(feature_matrix)
        else:
            # 连续值编码（4个参数通道）
            beam_params = self._parse_beam_folder(beam_folder)
            continuous_vector = self._create_continuous_encoding(beam_params)
            
            for i in range(4):
                param_channel = np.full((self.height, self.width), 
                                       continuous_vector[i], dtype=np.float32)
                inputs.append(param_channel)
        
        # 合并所有通道: [C, H, W]
        inputs = np.stack(inputs, axis=0).astype(np.float32)
        
        # 转换为张量
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(image_gain).unsqueeze(0)  # [1, H, W]
        valid_mask = torch.from_numpy(valid_area_mask.astype(np.float32))
        
        # 返回
        if self.mode == "dense":
            return inputs, targets, valid_mask
        else:
            # sparse模式需要生成采样掩码
            samples_mask = self._generate_sparse_samples(valid_area_mask, idx)
            samples_mask = torch.from_numpy(samples_mask).unsqueeze(0)  # [1, H, W]
            
            return inputs, targets, samples_mask, valid_mask


# ========================================================================
# 多稀疏度实验数据集
# ========================================================================

class RMEGANMultiSparsityDataset(RMEGANDataset):
    """
    支持多种稀疏度的数据集
    用于实验不同采样率对模型性能的影响
    """
    
    def __init__(self, 
                 sparsity_levels=[100, 200, 400, 819, 1000, 2000],
                 current_sparsity_idx=0,
                 **kwargs):
        """
        Args:
            sparsity_levels: 采样点数量列表
            current_sparsity_idx: 当前使用的稀疏度索引
            **kwargs: 传递给父类的参数
        """
        # 强制使用sparse模式
        kwargs['mode'] = 'sparse'
        kwargs['fix_samples'] = sparsity_levels[current_sparsity_idx]
        
        super().__init__(**kwargs)
        
        self.sparsity_levels = sparsity_levels
        self.current_sparsity_idx = current_sparsity_idx
        
        print(f"\n多稀疏度配置:")
        print(f"  可用稀疏度: {sparsity_levels}")
        print(f"  当前稀疏度: {sparsity_levels[current_sparsity_idx]} 采样点")
    
    def set_sparsity(self, sparsity_idx):
        """切换稀疏度级别"""
        if 0 <= sparsity_idx < len(self.sparsity_levels):
            self.current_sparsity_idx = sparsity_idx
            self.fix_samples = self.sparsity_levels[sparsity_idx]
            print(f"切换到稀疏度: {self.fix_samples} 采样点")
        else:
            raise ValueError(f"无效的稀疏度索引: {sparsity_idx}")
    
    def get_current_sparsity(self):
        """获取当前稀疏度"""
        return self.sparsity_levels[self.current_sparsity_idx]


# ========================================================================
# 测试代码
# ========================================================================

def test_dataset():
    """测试数据集"""
    print("=" * 80)
    print("测试 RME-GAN 数据集（改进版 v4 - 支持平面波/球面波）")
    print("=" * 80)
    
    # 修改为你的实际路径
    base_config = {
        "dir_multibeam": "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/multibeam_labeled_radiomaps",
        "dir_height_maps": "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/height_maps",
        "dir_feature_maps": "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/simulation_results_multibeam",
        "dir_feature_maps_plane": "/seu_share2/home/hanyu2/230258948/ogsp/Experiment/Beam_Map_PlaneWave",
        "use_feature_maps": True,
        "random_seed": 42
    }
    
    # 测试1: 球面波 Feature Map
    print("\n" + "-" * 40)
    print("测试1: 球面波 Feature Map + dense模式")
    print("-" * 40)
    
    try:
        dataset = RMEGANDataset(
            phase="train",
            split_strategy="random",
            mode="dense",
            wave_type="spherical",
            **base_config
        )
        
        inputs, targets, valid_mask = dataset[0]
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        print(f"  有效掩码形状: {valid_mask.shape}")
        print("  ✅ 通过")
    except FileNotFoundError as e:
        print(f"  ⚠️ 跳过（路径不存在）: {e}")
    
    # 测试2: 平面波 Feature Map
    print("\n" + "-" * 40)
    print("测试2: 平面波 Feature Map + sparse模式")
    print("-" * 40)
    
    try:
        dataset = RMEGANDataset(
            phase="train",
            split_strategy="random",
            mode="sparse",
            fix_samples=500,
            wave_type="plane",
            **base_config
        )
        
        inputs, targets, samples_mask, valid_mask = dataset[0]
        print(f"  输入形状: {inputs.shape}")
        print(f"  目标形状: {targets.shape}")
        print(f"  采样掩码形状: {samples_mask.shape}")
        print(f"  采样点数: {samples_mask.sum().item()}")
        print(f"  有效掩码形状: {valid_mask.shape}")
        print("  ✅ 通过")
    except FileNotFoundError as e:
        print(f"  ⚠️ 跳过（路径不存在）: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()