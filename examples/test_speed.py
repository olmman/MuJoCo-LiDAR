import time
from etils import epath

import mujoco
import numpy as np
from mujoco_lidar import MjLidarWrapper, scan_gen

np.set_printoptions(precision=3, suppress=True, linewidth=500)

# Load model
mjcf_file = epath.Path(__file__).parent.parent / "models" / "scene_primitive.xml"
mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
mj_data = mujoco.MjData(mj_model)
mujoco.mj_step(mj_model, mj_data)

# Generate scan pattern
theta, phi = scan_gen.generate_airy96()
print(f"Number of rays: {len(theta)}")

# Prepare random indices for sampling
np.random.seed(0)
rnd_args = np.random.randint(0, len(theta), size=30)

backends = ['cpu', 'taichi', 'jax']
results = {}

for backend in backends:
    print(f"\nInitializing {backend.upper()} LiDAR...")
    try:
        lidar = MjLidarWrapper(mj_model, site_name="lidar_site", backend=backend)
        
        # Warm up
        print("Running scan...")
        ranges = lidar.trace_rays(mj_data, theta, phi)
        if backend == 'jax':
            ranges.block_until_ready()
        
        # Timing
        start = time.time()
        num_runs = 10
        for _ in range(num_runs):
            ranges = lidar.trace_rays(mj_data, theta, phi)
            if backend == 'jax':
                ranges.block_until_ready()
        end = time.time()
        
        print(f"Scan time: {1e3 * (end - start) / num_runs:.2f}ms")
        
        # Store results
        ranges_np = np.array(ranges)
        ranges_sorted = np.sort(ranges_np)
        results[backend] = ranges_sorted[rnd_args]
        
    except Exception as e:
        print(f"Failed to run {backend} backend: {e}")

print("\n" + "="*120)
print("Summary of Sample Ranges:")
for backend in backends:
    if backend in results:
        print(f"{backend:<8}: {results[backend]}")
    else:
        print(f"{backend:<8}: Failed")
print("="*120)
