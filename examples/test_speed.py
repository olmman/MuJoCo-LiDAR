import argparse
import time
import mujoco
import numpy as np
import zhplot # 如果图像显示中文，需要安装zhplot
import matplotlib.pyplot as plt

# 尝试导入3D绘图模块，如果失败则使用2D替代
try:
    from mpl_toolkits.mplot3d import Axes3D  # 必需导入以支持3D绘图
    HAS_3D = True
except ImportError as e:
    print(f"Warning: 3D plotting not available due to matplotlib version conflict: {e}")
    print("Will use 2D projections instead.")
    HAS_3D = False

from mujoco_lidar import LidarSensor
from mujoco_lidar.scan_gen import generate_grid_scan_pattern
from mujoco_lidar.mj_lidar_utils import create_demo_scene

if __name__ == "__main__":
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR传感器演示和性能测试')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    parser.add_argument('--skip-test', action='store_true', help='跳过性能测试')
    parser.add_argument('--zh', action='store_true', help='使用中文标签')
    parser.add_argument('--save-fig', action='store_true', help='保存图表')
    args = parser.parse_args()

    use_english_labels = not args.zh
    
    # 创建MuJoCo场景
    mj_model, mj_data = create_demo_scene()
    # mj_model, mj_data = create_demo_scene("mesh_scene")
    
    # 创建场景渲染对象
    scene = mujoco.MjvScene(mj_model, maxgeom=10000)
    
    # 更新模拟
    mujoco.mj_forward(mj_model, mj_data)
    
    # 更新场景
    mujoco.mjv_updateScene(
        mj_model, mj_data, mujoco.MjvOption(), 
        None, mujoco.MjvCamera(), 
        mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )

    # 创建激光雷达传感器
    lidar = LidarSensor(mj_model, site_name="lidar_site")
    
    # 是否跳过性能测试
    if not args.skip_test:
        print("=" * 50)
        print("性能测试 - 不同射线数量")
        print("=" * 50)
        
        # 测试不同射线数量的性能
        ray_counts = [1000, 5000, 10000, 20000, 50000, 100000]
        total_times = []
        
        for count in ray_counts:
            print(f"\n测试射线数量: {count}")
            
            # 创建测试用的射线
            test_rays_phi, test_rays_theta = generate_grid_scan_pattern(
                num_ray_cols=int(np.sqrt(count) * 10), 
                num_ray_rows=int(np.sqrt(count) / 10)
            )

            # 优化内存布局
            test_rays_phi = np.ascontiguousarray(test_rays_phi[:count])
            test_rays_theta = np.ascontiguousarray(test_rays_theta[:count])
            
            # 执行多次测试取平均值
            n_tests = 5
            time_sum = 0
            
            # 预热
            lidar.update(mj_data, test_rays_phi, test_rays_theta)
            points = lidar.get_data_in_local_frame()
            
            for i in range(n_tests):
                # 执行光线追踪
                start_time = time.time()
                lidar.update(mj_data, test_rays_phi, test_rays_theta)
                points = lidar.get_data_in_local_frame()
                end_time = time.time()
               
                # 累加时间
                time_sum += (end_time - start_time) * 1000  # 转换为毫秒
            
            # 计算平均时间
            avg_time = time_sum / n_tests
            total_times.append(avg_time)
            
            print(f"平均处理时间: {avg_time:.2f}ms")
        
        # 绘制性能结果图表
        plt.figure(figsize=(12, 8))
        
        # 全局字体设置
        title_font = {'fontsize': 14, 'fontweight': 'bold'}
        label_font = {'fontsize': 12}
        
        # 根据字体可用性选择标签语言
        if use_english_labels:
            # 使用英文标签
            ray_count_title = 'Effect of Ray Count on Performance'
            ray_label = 'Ray Count'
            time_label = 'Time (ms)'
            total_time_legend = 'Total Time'
        else:
            # 使用中文标签
            ray_count_title = '射线数量对性能的影响'
            ray_label = '射线数量'
            time_label = '时间 (ms)'
            total_time_legend = '总时间'
        
        # 射线数量对性能的影响
        plt.plot(ray_counts, total_times, 'o-', label=total_time_legend, linewidth=2)
        plt.xlabel(ray_label, **label_font)
        plt.ylabel(time_label, **label_font)
        plt.title(ray_count_title, **title_font)
        plt.legend(prop={'size': 10})
        plt.grid(True)
        plt.tick_params(labelsize=10)
        
        plt.tight_layout()
        if args.save_fig:
            if use_english_labels:
                plt.savefig('lidar_performance_test_en.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig('lidar_performance_test_zh.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 将性能测试结果以表格形式打印
        print("\n" + "=" * 80)
        print("性能测试结果汇总")
        print("=" * 80)
        
        # 表格：射线数量对性能的影响
        print("\n表格: 射线数量对性能的影响")
        print("-" * 40)
        print(f"{'射线数量':^12} | {'总时间 (ms)':^15}")
        print("-" * 40)
        for i, count in enumerate(ray_counts):
            print(f"{count:^12} | {total_times[i]:^15.2f}")
        print("-" * 40)
    
    # 执行标准光线追踪测试
    print("\n执行标准光线追踪测试:")

    rays_theta, rays_phi = generate_grid_scan_pattern(num_ray_cols=1800, num_ray_rows=64)

    # 优化内存布局
    rays_phi = np.ascontiguousarray(rays_phi)
    rays_theta = np.ascontiguousarray(rays_theta)

    for _ in range(3):
        start_time = time.time()
        lidar.update(mj_data, rays_phi, rays_theta)
        points = lidar.get_data_in_local_frame()
        end_time = time.time()
    
    # 打印性能信息和当前位置
    print(f"耗时: {(end_time - start_time)*1000:.2f} ms, 射线数量: {len(rays_phi)}")
    
    # 三维点云可视化
    if HAS_3D:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)

        # 设置轴标签
        if use_english_labels:
            ax.set_xlabel('X-axis', fontsize=12)
            ax.set_ylabel('Y-axis', fontsize=12)
            ax.set_zlabel('Z-axis', fontsize=12)
            ax.set_title('Lidar 3D Point Cloud Visualization', fontsize=14, fontweight='bold')
            cbar.set_label('Height (Z-value)', fontsize=12)
        else:
            ax.set_xlabel('X轴', fontsize=12)
            ax.set_ylabel('Y轴', fontsize=12)
            ax.set_zlabel('Z轴', fontsize=12)
            ax.set_title('激光雷达三维点云可视化', fontsize=14, fontweight='bold')
            cbar.set_label('高度 (Z值)', fontsize=12)

        # 设置三轴等比例
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 设置网格线
        ax.grid(True)
        
        # 设置背景色为淡灰色，以更好地显示点云
        ax.set_facecolor((0.95, 0.95, 0.95))
        
        fig.tight_layout()
        if args.save_fig:
            if use_english_labels:
                plt.savefig('lidar_point_cloud_en.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig('lidar_point_cloud_zh.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        # 使用2D投影作为替代方案
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY平面投影
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap='viridis')
        if use_english_labels:
            ax1.set_xlabel('X-axis')
            ax1.set_ylabel('Y-axis')
            ax1.set_title('XY Plane Projection')
        else:
            ax1.set_xlabel('X轴')
            ax1.set_ylabel('Y轴')
            ax1.set_title('XY平面投影')
        ax1.grid(True)
        plt.colorbar(scatter1, ax=ax1)
        
        # XZ平面投影
        scatter2 = ax2.scatter(points[:, 0], points[:, 2], s=1, c=points[:, 1], cmap='viridis')
        if use_english_labels:
            ax2.set_xlabel('X-axis')
            ax2.set_ylabel('Z-axis')
            ax2.set_title('XZ Plane Projection')
        else:
            ax2.set_xlabel('X轴')
            ax2.set_ylabel('Z轴')
            ax2.set_title('XZ平面投影')
        ax2.grid(True)
        plt.colorbar(scatter2, ax=ax2)
        
        # YZ平面投影
        scatter3 = ax3.scatter(points[:, 1], points[:, 2], s=1, c=points[:, 0], cmap='viridis')
        if use_english_labels:
            ax3.set_xlabel('Y-axis')
            ax3.set_ylabel('Z-axis')
            ax3.set_title('YZ Plane Projection')
        else:
            ax3.set_xlabel('Y轴')
            ax3.set_ylabel('Z轴')
            ax3.set_title('YZ平面投影')
        ax3.grid(True)
        plt.colorbar(scatter3, ax=ax3)
        
        # 距离分布直方图
        distances = np.sqrt(np.sum(points**2, axis=1))
        ax4.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        if use_english_labels:
            ax4.set_xlabel('Distance from Origin')
            ax4.set_ylabel('Count')
            ax4.set_title('Distance Distribution')
        else:
            ax4.set_xlabel('距离原点距离')
            ax4.set_ylabel('数量')
            ax4.set_title('距离分布')
        ax4.grid(True)
        
        if use_english_labels:
            fig.suptitle('Lidar Point Cloud 2D Projections', fontsize=14, fontweight='bold')
        else:
            fig.suptitle('激光雷达点云二维投影', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if args.save_fig:
            if use_english_labels:
                plt.savefig('lidar_point_cloud_2d_en.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig('lidar_point_cloud_2d_zh.png', dpi=300, bbox_inches='tight')
        plt.show()