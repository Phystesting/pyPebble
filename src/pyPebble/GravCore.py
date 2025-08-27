import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from numba import cuda
import tqdm
import h5py
import math

# Constants
G = 0.0045  # Parsec^3 / (Msol * Megayears^2)

@cuda.jit(device=True)
def pair_accel(pos_i, pos_j, mass_i, mass_j, softening):
    G = 0.0045
    dx = pos_j[0] - pos_i[0]
    dy = pos_j[1] - pos_i[1]
    r = math.sqrt(dx*dx + dy*dy)
    r2 = r*r + softening
    F = G * mass_i * mass_j / r2

    dirx = dx / r
    diry = dy / r	
        
    # Apply force
    ax_i = (F * dirx) / mass_i
    ax_j = -(F * dirx) / mass_j
    ay_i = (F * diry) / mass_i
    ay_j = -(F * diry) / mass_j

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


class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass

class Bodies:
    def __init__(self, **kwargs):
        if "positions" in kwargs and "velocities" in kwargs and "masses" in kwargs:
            self.bodies = self.from_arrays(kwargs["positions"], kwargs["velocities"], kwargs["masses"])
        else:
            self.bodies = []

    def from_arrays(self, positions, velocities, masses):
        bodies = []
        n = len(positions)
        assert len(velocities) == n and len(masses) == n, "All arrays must have the same length"
        for i in range(n):
            bodies.append(Body(positions[i],velocities[i],masses[i]))
        return bodies
    def create_disc(self, n_particles, r_max, center, ang_vel, v_sigma=None, total_mass=None, particle_mass=None, distribution="uniform"):
        if total_mass:
            if particle_mass:
                ValueError("You cannot define both a particle mass and a total Mass!")
            particle_mass = total_mass / n_particles
        if particle_mass == None:
            ValueError("Please pass either particle mass or total mass.")
        positions = np.zeros([n_particles,2])
        velocities = np.zeros([n_particles,2])
        masses = np.full(n_particles, particle_mass)
        
        for i in range(n_particles):
            r = np.random.rand() * r_max
            theta = np.random.rand() * 2 * np.pi
            positions[i] = np.array([r * np.cos(theta), r * np.sin(theta)]) + np.array(center)
            vt = ang_vel * r
            vx = -vt * np.sin(theta) + (np.random.rand() - 0.5) * 2 * v_sigma
            vy =  vt * np.cos(theta) + (np.random.rand() - 0.5) * 2 * v_sigma
            velocities[i] = [vx, vy]
        self.bodies = self.from_arrays(positions, velocities, masses)
        return self
    
    def setup(self, **kwargs):
        if len(self.bodies) > 0:
            return Simulate(self, **kwargs)
        else:
            raise ValueError("No bodies defined! Please assign bodies before setting up the simulation.")

class Simulate:
    def __init__(self, bodies, softening=1e-2, bounds=20, smooth_len=10, 
                 t_start=0, t_finish=50, nsteps=1000, Enable_GPU=True, save_output=None):
        self.bodies = bodies.bodies
        self.softening = softening
        self.bounds = bounds
        self.smooth_len = smooth_len
        self.t_start = t_start
        self.t_finish = t_finish
        self.n_steps = nsteps
        self.Enable_GPU = Enable_GPU
        self.save_output = save_output
        self.dt = None
    
    def compute_accels(self, time, positions, velocities):
        masses = np.array([body.mass for body in self.bodies])
        accels = np.zeros_like(velocities)
        tree = cKDTree(positions)
        pairs = np.array(list(tree.query_pairs(r=self.smooth_len)), dtype=np.int32)
        if pairs.size == 0:
            return accels
        softening = self.softening
        if self.Enable_GPU:
            # Create a float32 host array to receive GPU results
            accels_gpu = np.zeros_like(velocities, dtype=np.float32)

            stream = cuda.stream()
            d_positions = cuda.to_device(positions.astype(np.float32), stream=stream)
            d_masses = cuda.to_device(masses.astype(np.float32), stream=stream)
            d_accels = cuda.to_device(accels_gpu, stream=stream)
            d_pairs = cuda.to_device(pairs, stream=stream)

            threadsperblock = 128
            blockspergrid = (len(pairs) + threadsperblock - 1) // threadsperblock

            compute_all_pairs[blockspergrid, threadsperblock, stream](d_pairs, d_positions, d_masses, d_accels, softening)

            # Copy into float32 array
            d_accels.copy_to_host(accels_gpu, stream=stream)
            stream.synchronize()

            # Convert to float64 for the rest of the code
            accels = accels_gpu.astype(np.float64)
        else:
            for i, j in pairs:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                direction = r_vec / r
                denom = r ** 2 + self.softening

                accels[i] += G * masses[j] * direction / denom
                accels[j] += -G * masses[i] * direction / denom

        return accels

    def rk4(self, time):
        positions = np.array([body.position for body in self.bodies])
        velocities = np.array([body.velocity for body in self.bodies])
        dt = self.dt

        k1_vel = self.compute_accels(time, positions, velocities)
        k1_pos = velocities

        k2_vel = self.compute_accels(time + 0.5 * dt, positions + 0.5 * dt * k1_pos, velocities + 0.5 * dt * k1_vel)
        k2_pos = velocities + 0.5 * dt * k1_vel

        k3_vel = self.compute_accels(time + 0.5 * dt, positions + 0.5 * dt * k2_pos, velocities + 0.5 * dt * k2_vel)
        k3_pos = velocities + 0.5 * dt * k2_vel

        k4_vel = self.compute_accels(time + dt, positions + dt * k3_pos, velocities + dt * k3_vel)
        k4_pos = velocities + dt * k3_vel

        dpos = dt * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
        dvel = dt * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6

        for i, body in enumerate(self.bodies):
            body.position += dpos[i]
            body.velocity += dvel[i]

    def periodic_boundaries(self):
        for body in self.bodies:
            body.position = np.mod(body.position + self.bounds, 2 * self.bounds) - self.bounds

    def run(self):
        n_bodies = len(self.bodies)
        t_vals = np.linspace(self.t_start, self.t_finish, self.n_steps)
        self.dt = t_vals[1] - t_vals[0]
        if self.save_output:
            with h5py.File(self.save_output, "w") as f:
                dset_time = f.create_dataset("time",shape=(self.n_steps), dtype='f8')
                dset_pos = f.create_dataset("positions",shape=(self.n_steps,n_bodies,2), dtype='f8')
                dset_vel = f.create_dataset("velocities",shape=(self.n_steps,n_bodies,2), dtype='f8')
                dset_mass = f.create_dataset("masses",shape=(self.n_steps,n_bodies), dtype='f8')
                for step, t in enumerate(tqdm.tqdm(t_vals, desc="Running simulation")):
                    self.periodic_boundaries()
                    self.rk4(t)
                    
                    positions = np.array([body.position for body in self.bodies])
                    velocities = np.array([body.velocity for body in self.bodies])
                    masses = np.array([body.mass for body in self.bodies])
                    
                    dset_time[step] = t
                    dset_pos[step] = positions
                    dset_vel[step] = velocities
                    dset_mass[step] = masses
                    


