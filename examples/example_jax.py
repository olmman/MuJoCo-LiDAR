import time
import mujoco
import numpy as np
from etils import epath
from mujoco_lidar import MjLidarWrapper, scan_gen

np.set_printoptions(precision=3, suppress=True, linewidth=200)

# Load model
mjcf_file = epath.Path(__file__).parent.parent / "models" / "scene_jax.xml"
mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
mj_data = mujoco.MjData(mj_model)
mujoco.mj_step(mj_model, mj_data)

# Initialize LiDAR with JAX backend
print("Initializing JAX LiDAR...")
lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend="jax")

# Generate scan pattern
generator = scan_gen.LivoxGenerator("mid360")
theta, phi = generator.sample_ray_angles()
print(f"Number of rays: {len(theta)}")

# Run scan
print("Running scan...")
ranges = lidar.trace_rays(mj_data, theta, phi)

start = time.time()
num_runs = 10
for _ in range(num_runs):
    ranges = lidar.trace_rays(mj_data, theta, phi)
end = time.time()

print(f"Scan time: {1e3 * (end - start) / num_runs:.2f}ms")
print(f"Output shape: {ranges.shape}")
ranges = np.sort(ranges)
print(f"Sample ranges: {ranges[-10:]}")