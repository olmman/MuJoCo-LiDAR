import os
import numpy as np
from functools import lru_cache
import taichi as ti

ti.init(arch=ti.gpu)

@ti.data_oriented
class LivoxGenerator:
    """Livox 扫描模式：预加载全部角度到 Taichi，采样返回 Taichi ndarray。"""
    livox_lidar_params = {
        "avia": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 70.4, "vertical_fov": 77.2, "samples": 24000},
        "HAP": {"laser_min_range": 0.1, "laser_max_range": 200.0, "samples": 45300, "downsample": 1},
        "horizon": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000},
        "mid40": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 81.7, "vertical_fov": 25.1, "samples": 24000},
        "mid70": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 70.4, "vertical_fov": 70.4, "samples": 10000},
        "mid360": {"laser_min_range": 0.1, "laser_max_range": 200.0, "samples": 24000},
        "tele": {"laser_min_range": 0.1, "laser_max_range": 200.0, "horizontal_fov": 14.5, "vertical_fov": 16.1, "samples": 24000},
    }

    def __init__(self, name: str):
        if name not in self.livox_lidar_params:
            raise ValueError(f"Invalid LiDAR name: {name}")
        p = self.livox_lidar_params[name]
        self.samples = p["samples"]
        pattern_npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_mode", f"{name}.npy")
        if not os.path.isfile(pattern_npy_path):
            raise FileNotFoundError(f"Scan mode file not found: {pattern_npy_path}")
        ray_angles_np = np.load(pattern_npy_path).astype(np.float32)  # shape (N,2) -> (theta, phi)
        if ray_angles_np.shape[1] != 2:
            raise ValueError("scan_mode npy 第二维应为2 (theta, phi)")
        self.n_rays = ray_angles_np.shape[0]
        # 全量角度放入 field
        self.theta_all = ti.field(dtype=ti.f32, shape=self.n_rays)
        self.phi_all = ti.field(dtype=ti.f32, shape=self.n_rays)
        self.theta_all.from_numpy(ray_angles_np[:, 0])
        self.phi_all.from_numpy(ray_angles_np[:, 1])
        # 采样缓存（可重建）
        self._theta_sample = None
        self._phi_sample = None
        self._sample_size = 0
        self.currStartIndex = 0

    def _ensure_sample_buf(self, size: int):
        if self._theta_sample is None or self._sample_size != size:
            self._theta_sample = ti.ndarray(dtype=ti.f32, shape=size)
            self._phi_sample = ti.ndarray(dtype=ti.f32, shape=size)
            self._sample_size = size

    @ti.kernel
    def _gather_kernel(self, 
                       start: ti.i32, 
                       step: ti.i32, 
                       size: ti.i32, 
                       theta_out: ti.types.ndarray(dtype=ti.f32, ndim=1), 
                       phi_out: ti.types.ndarray(dtype=ti.f32, ndim=1), 
                       n_total: ti.i32
    ):
        for i in ti.ndrange(size):
            idx = (start + i * step) % n_total
            theta_out[i] = self.theta_all[idx]
            phi_out[i] = self.phi_all[idx]

    def sample_ray_angles_ti(self, downsample: int = 1):
        if downsample < 1:
            downsample = 1
        eff = self.samples // downsample if downsample > 1 else self.samples
        self._ensure_sample_buf(eff)
        self._gather_kernel(self.currStartIndex, downsample, eff, self._theta_sample, self._phi_sample, self.n_rays)
        # 前进“samples”步（保持与原算法一致）
        self.currStartIndex = (self.currStartIndex + self.samples) % self.n_rays
        return self._theta_sample, self._phi_sample

    # 兼容旧接口（需要时仍可得到 numpy，但会有拷贝）
    def sample_ray_angles(self, downsample=1):
        th_ti, ph_ti = self.sample_ray_angles_ti(downsample)
        return th_ti.to_numpy(), ph_ti.to_numpy()

# =======================================================================
# 1. Velodyne HDL-64 (任意 360° 旋转式激光雷达)
# =======================================================================
def generate_HDL64(     # |参数            | Velodyne HDL-64
    f_rot=10.0,            # |转速 (Hz)       |  5-20Hz
    sample_rate=1.1e6,     # |采样率 (Hz)     | 2.2MHz(双返回模式)
    n_channels=64,         # |垂直通道数       | 64 (Vertical Angular Resolution : 0.4°)
    phi_fov=(-24.9, 2.)    # |垂直视场角 (度)  | (-24.9°, 2.°)
):
    # 转换为弧度
    phi_min, phi_max = np.deg2rad(phi_fov)
    
    # 时间序列（列向量）
    t = np.arange(0, 1./f_rot, n_channels/sample_rate)[:, None]  # shape: (n_times, 1)
    
    # 水平角计算（广播机制）
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)      # shape: (n_times, 1)
    
    # 垂直角（行向量）
    phi = np.linspace(phi_min, phi_max, n_channels)     # shape: (1, n_channels)
    
    # 生成网格（无需显式使用meshgrid）
    theta_grid = theta + np.zeros((1, n_channels))      # 广播至 (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi               # 广播至 (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 2. Velodyne VLP-32 模式
# https://www.mapix.com/lidar-scanner-sensors/velodyne/velodyne-vlp-32c/
# =======================================================================
@lru_cache(maxsize=8)
def _get_vlp32_angles():
    """使用缓存获取VLP-32的角度分布，避免重复计算，返回弧度值"""
    vlp32_angles = np.array([
        -25.0, -22.5, -20.0, -15.0, -13.0, -10.0, -5.0, -3.0, 
        -2.333, -1.0, -0.667, -0.333, 0.0, 0.0, 0.333, 0.667, 
        1.0, 1.333, 1.667, 2.0, 2.333, 2.667, 3.0, 3.333, 
        3.667, 4.0, 5.0, 7.0, 10.0, 15.0, 17.0, 20.0
    ])
    # 转换为弧度并裁剪
    vlp32_angles = np.deg2rad(vlp32_angles)
    return vlp32_angles

def generate_vlp32(
    f_rot=10.0,       # 转速 (Hz)
    sample_rate=1.2e6 # 采样率 (Hz)
):
    # 垂直角参数
    phi = _get_vlp32_angles()       # shape: (n_channels,)
    
    # 时间序列（列向量）
    t = np.arange(0, 1/f_rot, 32/sample_rate)[:, None]  # shape: (n_times, 1)
    
    # 水平角计算
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)      # shape: (n_times, 1)
    
    # 广播生成网格
    theta_grid = theta + np.zeros_like(phi)            # shape: (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi              # shape: (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 3. Ouster OS-128 模式
# https://www.general-laser.at/en/shop-en/ouster-os0-128-lidar-sensor-en
# =======================================================================
def generate_os128(
    f_rot=20.0,            # 转速 (Hz)
    sample_rate=5.2e6,     # 采样率 (Hz)
):
    # 垂直角参数（均匀分布）
    n_channels = 128
    phi = np.deg2rad(np.linspace(-22.5, 22.5, n_channels))  # shape: (n_channels,)
    
    # 时间序列（列向量）
    t = np.arange(0, 1/f_rot, n_channels/sample_rate)[:, None]     # shape: (n_times, 1)
    
    # 水平角计算
    theta = (2 * np.pi * f_rot * t) % (2 * np.pi)         # shape: (n_times, 1)
    
    # 广播生成网格
    theta_grid = theta + np.zeros_like(phi)               # shape: (n_times, n_channels)
    phi_grid = np.zeros_like(theta) + phi                 # shape: (n_times, n_channels)
    
    return theta_grid.flatten(), phi_grid.flatten()

# =======================================================================
# 4. 生成网格状扫描模式
# =======================================================================
def generate_grid_scan_pattern(num_ray_cols, num_ray_rows, theta_range=(-np.pi, np.pi), phi_range=(-np.pi/3, np.pi/3)):
    """
    生成网格状扫描模式
    
    参数:
        num_ray_cols: 水平方向射线数
        num_ray_rows: 垂直方向射线数
        
    返回:
        (ray_theta, ray_phi): 水平角和垂直角数组
    """
   # 创建网格扫描模式
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(theta_range[0], theta_range[1], num_ray_cols),  # 水平角
        np.linspace(phi_range[0], phi_range[1], num_ray_rows)  # 垂直角
    )
    
    # 展平网格为一维数组
    ray_phi = phi_grid.flatten()
    ray_theta = theta_grid.flatten()
    
    # 打印扫描范围信息
    print(f"扫描模式：phi范围[{ray_phi.min():.2f}, {ray_phi.max():.2f}], theta范围[{ray_theta.min():.2f}, {ray_theta.max():.2f}]")
    return ray_theta, ray_phi

# =======================================================================
# 5. 创建激光雷达扫描线的角度数组，仅包含水平方向
# =======================================================================
def create_lidar_single_line(horizontal_resolution=360, horizontal_fov=2*np.pi):
    """创建激光雷达扫描线的角度数组，仅包含水平方向"""
    h_angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, horizontal_resolution)
    v_angles = np.zeros_like(h_angles)
    return h_angles, v_angles
