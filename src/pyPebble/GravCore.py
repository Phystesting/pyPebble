import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
import tqdm
import h5py
import math
import multiprocessing as mp
from .gpu_module import gpu_accels
import contextlib

# Constants
G = 0.0045  # Parsec^3 / (Msol * Megayears^2)

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass

class Pebbles:
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
    def __init__(self, bodies, softening=1, bounds=20, smooth_len=30, t_start=0, t_finish=50, n_steps=1000, Enable_GPU=True, save_output=None):
        self.bodies = bodies.bodies
        self.softening = softening
        self.bounds = bounds
        self.smooth_len = smooth_len
        self.t_start = t_start
        self.t_finish = t_finish
        self.n_steps = n_steps
        self.n_bodies = len(bodies.bodies)
        self.dt = None
        
        self.Enable_GPU = Enable_GPU
        self.save_output = save_output
        
        self.manager = mp.Manager()
        self.shared_state = self.manager.dict()
        self.shared_state["positions"] = np.array([body.position for body in bodies.bodies])
        self.shared_state["velocities"] =  np.array([body.velocity for body in bodies.bodies])
        self.shared_state["time"] = 0
        self.lock = self.manager.Lock()
        self.anim_process = None
    
    def compute_accels(self, time, positions, velocities):
        masses = np.array([body.mass for body in self.bodies])
        accels = np.zeros_like(velocities)
        tree = cKDTree(positions)
        pairs = np.array(list(tree.query_pairs(r=self.smooth_len)), dtype=np.int32)
        softening = self.softening
        if pairs.size == 0:
            return accels
        if self.Enable_GPU:
            accels = gpu_accels(pairs,positions,velocities,masses,softening)
        else:
            for i, j in pairs:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                direction = r_vec / r
                denom = r ** 2 + self.softening

                accels[i] += G * masses[j] * direction / denom
                accels[j] += -G * masses[i] * direction / denom

        return accels

    def rk4_step(self, time):
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
        self.dt = (self.t_finish - self.t_start) / (self.n_steps - 1)
        t_vals = np.linspace(self.t_start, self.t_finish, self.n_steps)
        with self._setup_file() as (dset_time,dset_pos,dset_vel,dset_mass):
            for step, t in enumerate(tqdm.tqdm(t_vals, desc="Running simulation")):
                self.periodic_boundaries()
                self.rk4_step(t)
                    
                positions = np.array([body.position for body in self.bodies])
                velocities = np.array([body.velocity for body in self.bodies])
                masses = np.array([body.mass for body in self.bodies])
                
                with self.lock:
                    self.shared_state["positions"] = positions
                    self.shared_state["velocities"] = velocities
                    self.shared_state["time"] = t
                 
                if dset_time is not None:
                    dset_time[step] = t
                    dset_pos[step] = positions
                    dset_vel[step] = velocities
                    dset_mass[step] = masses
                
                
                
        if self.anim_process is not None and self.anim_process.is_alive():
            self.stop_animation()
        return self
    
    @contextlib.contextmanager
    def _setup_file(self):
        if self.save_output:
            f = h5py.File(self.save_output, "w")
            dset_time = f.create_dataset("time",shape=(self.n_steps), dtype='f8')
            dset_pos = f.create_dataset("positions",shape=(self.n_steps,self.n_bodies,2), dtype='f8')
            dset_vel = f.create_dataset("velocities",shape=(self.n_steps,self.n_bodies,2), dtype='f8')
            dset_mass = f.create_dataset("masses",shape=(self.n_steps,self.n_bodies), dtype='f8')
            yield dset_time, dset_pos, dset_vel, dset_mass
            f.close()
        else:
            yield None, None, None, None
        
    def start_animation(self):
        if self.anim_process is None or not self.anim_process.is_alive():
            self.anim_process = mp.Process(target=self._run_ani)
            self.anim_process.start()
        return self
        
    def stop_animation(self):
        if self.anim_process is not None:
            if self.anim_process.is_alive():
                self.anim_process.terminate()
                self.anim_process.join()
            self.anim_process = None
        return self
    def _run_ani(self):
        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=5)
        ax.set_xlim(-self.bounds, self.bounds)
        ax.set_ylim(-self.bounds, self.bounds)

        def update(frame):
            with self.lock:
                pos = self.shared_state["positions"].copy()
                t = self.shared_state["time"]
            scat.set_offsets(pos)
            ax.set_title(f"t = {t:.2f}")
            return scat,

        ani = FuncAnimation(fig, update, interval=50, blit=False)
        plt.show()
            








