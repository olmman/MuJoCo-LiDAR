import time
import argparse
import traceback
import numpy as np
import os

import mujoco
import mujoco.viewer
import taichi as ti
from scipy.spatial.transform import Rotation

# 使用Open3D进行可视化
import open3d as o3d

from mujoco_lidar import (
    LidarSensor, LivoxGenerator, 
    generate_vlp32, generate_HDL64, generate_os128
)
from mujoco_lidar.mj_lidar_utils import create_demo_scene, KeyboardListener


class SimpleLidarVisualizer:
    """简化版实时LiDAR可视化器 (已更新为使用新的 LidarSensor 接口)"""
    
    def __init__(self, args):
        # 创建MuJoCo场景
        self.mj_model, self.mj_data = create_demo_scene("floor")
        
        # 设置LiDAR类型
        self.use_livox_lidar = False
        if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = LivoxGenerator(args.lidar)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif args.lidar == "HDL64":
            self.rays_theta, self.rays_phi = generate_HDL64()
        elif args.lidar == "vlp32":
            self.rays_theta, self.rays_phi = generate_vlp32()
        elif args.lidar == "os128":
            self.rays_theta, self.rays_phi = generate_os128()
        else:
            raise ValueError(f"不支持的LiDAR型号: {args.lidar}")

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)
        
        # 选择 OBJ (仅 GPU 后端需要)
        if args.backend == "gpu":
            if args.obj_path and os.path.exists(args.obj_path):
                obj_path = args.obj_path
            else:
                obj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models", "scene.obj")
        else:
            obj_path = None  # CPU 不使用
        
        # 创建激光雷达传感器 (新的统一接口)
        self.lidar = LidarSensor(
            self.mj_model,
            site_name="lidar_site",
            backend=args.backend,
            obj_path=obj_path
        )
        
        # 如果是 GPU，需要把角度转成 taichi ndarray
        n_rays = len(self.rays_theta)
        if self.lidar.backend == "gpu":
            _rays_phi = ti.ndarray(dtype=ti.f32, shape=n_rays)
            _rays_theta = ti.ndarray(dtype=ti.f32, shape=n_rays)
            _rays_phi.from_numpy(self.rays_phi)
            _rays_theta.from_numpy(self.rays_theta)
            self.rays_phi = _rays_phi
            self.rays_theta = _rays_theta
        
        print(f"射线数量: {n_rays}")
        
        # 获取激光雷达初始位置和方向
        lidar_base_position = self.mj_model.body("lidar_base").pos
        lidar_base_orientation = self.mj_model.body("lidar_base").quat[[1,2,3,0]]
        
        # 创建键盘监听器
        self.kb_listener = KeyboardListener(lidar_base_position, lidar_base_orientation)
        
        # 配置参数
        self.args = args
        self.running = True
        
        # 创建可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="MuJoCo LiDAR", width=1280, height=720)

        self.setup_rendering()
        
        # 创建点云对象
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

    def setup_rendering(self):
        """设置渲染参数"""
        # 获取渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # 深灰色背景
        opt.point_size = 10.0
        opt.show_coordinate_frame = True
        
        # 设置视角
        ctr = self.vis.get_view_control()
        ctr.set_front([0.5, 0.5, 0.5])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.1)  # 进一步缩小以看到更大范围

    def distance_colormap(self, points: np.ndarray) -> np.ndarray:
        """根据距离生成颜色映射"""
        # 计算距离
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.percentile(distances, 95) if len(points) > 0 else 1.0
        normalized = np.clip(distances / max_dist, 0, 1)
        colors = np.zeros((len(points), 3), dtype=np.float32)

        # 蓝色到绿色 (0-0.5)
        mask1 = normalized <= 0.5
        colors[mask1, 1] = normalized[mask1] * 2  # 绿色分量增加
        colors[mask1, 2] = 1 - normalized[mask1] * 2  # 蓝色分量减少
        
        # 绿色到红色 (0.5-1)
        mask2 = ~mask1
        colors[mask2, 0] = (normalized[mask2] - 0.5) * 2  # 红色分量增加
        colors[mask2, 1] = 1 - (normalized[mask2] - 0.5) * 2  # 绿色分量减少
        
        return colors
    
    def filter_point_cloud(self, points: np.ndarray, max_distance: float = 100.0, min_distance: float = 0.1) -> np.ndarray:
        """过滤点云，移除距离过远或过近的点"""
        if len(points) == 0:
            return points
            
        # 计算每个点到原点的距离
        distances = np.linalg.norm(points, axis=1)
        
        # 过滤条件：距离在合理范围内，且不是NaN或无穷大
        valid_mask = (
            (distances >= min_distance) & 
            (distances <= max_distance) & 
            np.isfinite(distances) &
            np.isfinite(points).all(axis=1)
        )
        
        filtered_points = points[valid_mask]
        
        # 只在verbose模式下打印过滤信息
        if self.args.verbose:
            print(f"点云过滤: {len(points)} -> {len(filtered_points)} 点")
        
        return filtered_points
    
    def update_visualization(self, points: np.ndarray):
        """更新可视化内容"""
        # 转换点云格式
        if points.ndim == 2 and points.shape[0] == 3 and points.shape[1] != 3:
            points = points.T
        
        # 过滤掉无效点（距离过远或过近的点）
        points = self.filter_point_cloud(points, self.args.max_distance, self.args.min_distance)
        
        # 更新点云
        # center = np.mean(points, axis=0)
        # points = points - center  # 平移到原点
        # max_dist = np.linalg.norm(points, axis=1).max()
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.distance_colormap(points))

        # 更新几何体
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def _resample_livox_if_needed(self):
        """根据后端重新采样 Livox 角度 (保持与 ros2 示例一致)"""
        if not self.use_livox_lidar:
            return
        if self.lidar.backend == "cpu":
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            # 保持 numpy contiguous
            self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
            self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)
        else:  # gpu
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles_ti()
    
    def run(self):
        """启动仿真和可视化"""
        step_cnt = 0
        step_gap = max(1, 60 // self.args.rate)
        
        try:
            with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
                # 设置视图模式
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value
                
                print("\n" + "=" * 60)
                print("控制说明: WASD/QE 移动, 方向键旋转, ESC 退出")
                print("=" * 60)
                
                while (self.running and self.kb_listener.running and 
                       viewer.is_running and self.vis.poll_events()):
                    
                    # 更新激光雷达位置和方向
                    site_position, site_orientation = self.kb_listener.update_lidar_pose(1./60.)
                    self.mj_model.body("lidar_base").pos[:] = site_position[:]
                    self.mj_model.body("lidar_base").quat[:] = site_orientation[[3,0,1,2]]
                    
                    # 更新仿真
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    step_cnt += 1
                    viewer.sync()
                    
                    # 按频率更新点云
                    if step_cnt % step_gap == 0:
                        start = time.time()
                        # Livox 需要每帧（或按频率）重新采样
                        self._resample_livox_if_needed()
                        # 执行 ray casting (统一接口)
                        self.lidar.update(self.mj_data, self.rays_phi, self.rays_theta)
                        # 获取点云 (局部) 并转世界
                        # pts_world = self.lidar.get_data_in_world_frame()
                        pts_world = self.lidar.get_data_in_local_frame()
                        self.update_visualization(pts_world)
                        end = time.time()
                        if self.args.verbose:
                            quat = Rotation.from_matrix(self.lidar.sensor_rotation).as_quat()
                            euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                            print(f"位置: [{self.lidar.sensor_position[0]:.2f},{self.lidar.sensor_position[1]:.2f},{self.lidar.sensor_position[2]:.2f}] "+
                                  f"范围: x=({pts_world[:,0].min():.2f} {pts_world[:,0].max():.2f}), y=({pts_world[:,1].min():.2f} {pts_world[:,1].max():.2f}), z=({pts_world[:,2].min():.2f} {pts_world[:,2].max():.2f}) "+
                                  f"欧拉: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°] "
                                  f"点数: {pts_world.shape[0]} 耗时: {(end-start)*1000:.2f}ms")
                    time.sleep(1./60.)  # 60Hz仿真频率
                    
        except KeyboardInterrupt:
            print("用户中断，正在退出...")
        except Exception as e:
            print(f"仿真出错: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            self.vis.destroy_window()
            np.save("mesh_test_hit_points.npy", pts_world)

def main():
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR简化可视化（不依赖ROS）- 已更新接口')
    parser.add_argument('--lidar', type=str, default='mid360',
                        choices=['avia', 'HAP', 'horizon', 'mid40', 'mid70', 'mid360', 'tele', 'HDL64', 'vlp32', 'os128'])
    parser.add_argument('--backend', type=str, default='gpu', choices=['cpu', 'gpu'], help='LiDAR后端 (cpu/gpu)')
    parser.add_argument('--obj-path', type=str, help='GPU模式下用于构建BVH的OBJ路径 (默认使用models/scene.obj)')
    parser.add_argument('--profiling', action='store_true', help='(保留参数, 当前接口未实现详细profiling输出)')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=30, help='LiDAR更新频率Hz')
    parser.add_argument('--max_distance', type=float, default=500.0)
    parser.add_argument('--min_distance', type=float, default=0.05)
    args = parser.parse_args()
    print("\n" + "=" * 70)
    print("MuJoCo LiDAR简化可视化 (Open3D, 新LidarSensor接口)")
    print("=" * 70)
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- 后端: {args.backend}")
    if args.backend == 'gpu':
        print(f"- OBJ: {args.obj_path if args.obj_path else '默认 scene.obj'}")
    print(f"- 更新频率: {args.rate} Hz")
    print(f"- 可视化范围: {args.min_distance}-{args.max_distance} m")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")
    print("=" * 70)
    try:
        app = SimpleLidarVisualizer(args)
        app.run()
    except KeyboardInterrupt:
        print("用户中断，程序退出")
    except Exception as e:
        print(f"程序出错: {e}")
        traceback.print_exc()
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()