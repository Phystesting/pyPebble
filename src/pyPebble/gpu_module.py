import math
import numpy as np
try:
    from numba import cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
if GPU_AVAILABLE:
    @cuda.jit(device=True)
    def pair_accel(pos_i, pos_j, mass_i, mass_j, softening):
        G = 0.0045
        dx = pos_j[0] - pos_i[0]
        dy = pos_j[1] - pos_i[1]
        r = math.sqrt(dx*dx + dy*dy)
        r2 = r*r + softening

        dirx = dx / r
        diry = dy / r	
        
        # i feels acceleration from j
        ax_i = G * mass_j * dirx / r2
        ay_i = G * mass_j * diry / r2

        # j feels acceleration from i
        ax_j = -G * mass_i * dirx / r2
        ay_j = -G * mass_i * diry / r2

        return ax_i, ay_i, ax_j, ay_j

    @cuda.jit
    def compute_all_pairs(pair_indices, positions, masses, accels, softening):
        idx = cuda.grid(1)
        if idx < pair_indices.shape[0]:
            i = pair_indices[idx, 0]
            j = pair_indices[idx, 1]
            ax_i, ay_i, ax_j, ay_j = pair_accel(positions[i], positions[j], masses[i], masses[j], softening)
            cuda.atomic.add(accels, (i, 0), ax_i)
            cuda.atomic.add(accels, (i, 1), ay_i)
            cuda.atomic.add(accels, (j, 0), ax_j)
            cuda.atomic.add(accels, (j, 1), ay_j)


    def gpu_accels(pairs,positions,velocities,masses,softening):
        accels_gpu = np.zeros_like(velocities, dtype=np.float32)

        stream = cuda.stream()
        d_positions = cuda.to_device(positions.astype(np.float32), stream=stream)
        d_masses = cuda.to_device(masses.astype(np.float32), stream=stream)
        d_accels = cuda.to_device(accels_gpu, stream=stream)
        d_pairs = cuda.to_device(pairs, stream=stream)

        threadsperblock = 128
        blockspergrid = (len(pairs) + threadsperblock - 1) // threadsperblock

        compute_all_pairs[blockspergrid, threadsperblock, stream](d_pairs, d_positions, d_masses, d_accels, softening)

        d_accels.copy_to_host(accels_gpu, stream=stream)
        stream.synchronize()

        accels = accels_gpu.astype(np.float64)
        return accels

else:
    def gpu_accels(*args, **kwargs):
        raise RuntimeError("GPU acceleration requires Numba with CUDA support. Please ensure you have a NVIDIA GPU and the CUDA toolkit installed.")

