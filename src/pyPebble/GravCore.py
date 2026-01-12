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
class Constants:
    def __init__(self, unit_system):
        if unit_system == "psm":
            self.psm()
        elif unit_system == "asy":
            self.asy()
        else:
            raise ValueError("Unknown unit system")
    def psm(self):
        self.G = 0.0045  # Parsec^3 / (Msol * Megayears^2)
    def asy(self):
        self.G = 39.4784176  # AU^3 / (Msol * Years^2)
class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        
    def copy(self):
        return Body(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass
        )

class Pebbles:
    def __init__(self, **kwargs):
        if "positions" in kwargs and "velocities" in kwargs and "masses" in kwargs:
            self.bodies = self.from_arrays(kwargs["positions"], kwargs["velocities"], kwargs["masses"])
        else:
            self.bodies = []
    
    def __iter__(self):
        return iter(self.bodies)
    
    def __len__(self):
        return len(self.bodies)
    
    @property
    def positions(self):
        return np.array([b.position for b in self.bodies])

    @property
    def velocities(self):
        return np.array([b.velocity for b in self.bodies])

    @property
    def masses(self):
        return np.array([b.mass for b in self.bodies])
    
    def copy(self):
        new = Pebbles()
        # clone bodies/masses/arrays so they don’t share memory
        new.bodies = [body.copy() for body in self.bodies]
        return new
    
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
    def __init__(self, pebbles, units="asy", softening=1, bounds=20, cutoff_len=30, t_start=0, t_finish=50, n_steps=1000, Enable_GPU=True, save_output=None, track_energy=False, energy_samples=20):
        self.constants = Constants(units)
        self.G = self.constants.G
        self.pebbles = pebbles
        self.softening = softening
        self.bounds = bounds
        self.cutoff_len = cutoff_len
        self.t_start = t_start
        self.t_finish = t_finish
        self.n_steps = n_steps
        self.n_bodies = len(pebbles)
        self.dt = None
        self.energy_samples=energy_samples
        
        self.track_energy = track_energy
        self.Enable_GPU = Enable_GPU
        self.save_output = save_output
        
        self.manager = mp.Manager()
        self.shared_state = self.manager.dict()
        self.shared_state["positions"] = pebbles.positions
        self.shared_state["velocities"] =  pebbles.velocities
        self.shared_state["time"] = 0
        self.shared_state["KE"] = 0
        self.shared_state["U"] = 0
        self.shared_state["E"] = 0
        self.lock = self.manager.Lock()
        self.anim_process = None
    def compute_potential(self):
        positions = self.pebbles.positions
        masses = self.pebbles.masses
        U = 0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r = np.linalg.norm(positions[j] - positions[i])
                U -= self.G * masses[i] * masses[j] / np.sqrt(r**2 + self.softening)
        return U
        
    def compute_kinetic(self):
        velocities = self.pebbles.velocities
        masses = self.pebbles.masses
        return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        
        
    def compute_accels(self, time, positions, velocities):
        masses = self.pebbles.masses
        accels = np.zeros_like(velocities)
        tree = cKDTree(positions)
        cutoff_len = np.sqrt(max(masses))
        pairs = np.array(list(tree.query_pairs(r=self.cutoff_len)), dtype=np.int32)
        softening = self.softening
        if pairs.size == 0:
            return accels
        if self.Enable_GPU:
            accels = gpu_accels(self.constants, pairs,positions,velocities,masses,softening)
        else:
            for i, j in pairs:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                direction = r_vec / r
                denom = r ** 2 + self.softening

                accels[i] += self.G * masses[j] * direction / denom
                accels[j] += -self.G * masses[i] * direction / denom

        return accels

    def rk4_step(self, time):
        positions = self.pebbles.positions
        velocities = self.pebbles.velocities
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

        for i, body in enumerate(self.pebbles):
            body.position += dpos[i]
            body.velocity += dvel[i]

    def periodic_boundaries(self):
        for body in self.pebbles:
            body.position = np.mod(body.position + self.bounds, 2 * self.bounds) - self.bounds

    def run(self):
        self.dt = (self.t_finish - self.t_start) / (self.n_steps - 1)
        t_vals = np.linspace(self.t_start, self.t_finish, self.n_steps)
        check_energy = set(np.linspace(0, self.n_steps - 1, self.energy_samples, dtype=int))
        with self._setup_file() as (dset_time,dset_pos,dset_vel,dset_mass):
            for step, t in enumerate(tqdm.tqdm(t_vals, desc="Running simulation")):
                self.periodic_boundaries()
                self.rk4_step(t)
                    
                positions = self.pebbles.positions
                velocities = self.pebbles.velocities
                masses = self.pebbles.masses
                if self.track_energy and step in check_energy:
                    U = self.compute_potential()
                    KE = self.compute_kinetic()
                    E = KE + U
                with self.lock:
                    self.shared_state["positions"] = positions
                    self.shared_state["velocities"] = velocities
                    self.shared_state["time"] = t
                    if self.track_energy and step in check_energy:
                        self.shared_state["U"] = U
                        self.shared_state["KE"] = KE
                        self.shared_state["E"] = E
                 
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
        if self.track_energy:
            fig, (ax_pos, ax_energy) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [2, 1]})
        else:
            fig, ax_pos = plt.subplots(figsize=(6, 6))
            ax_energy = None  # no energy plot
        
        # Scatter plot for positions
        scat = ax_pos.scatter([], [], s=5)
        ax_pos.set_xlim(-self.bounds, self.bounds)
        ax_pos.set_ylim(-self.bounds, self.bounds)
        ax_pos.set_title("Particle Positions")
        
        # Prepare energy plot if enabled
        if self.track_energy:
            t_history = []
            KE_history = []
            U_history = []
            E_history = []

            line_KE, = ax_energy.plot([], [], label="KE")
            line_U, = ax_energy.plot([], [], label="U")
            line_E, = ax_energy.plot([], [], label="Total E")
            ax_energy.set_xlim(0, self.t_finish)
            ax_energy.set_ylim(-1, 1) 
            ax_energy.set_ylabel("Energy")
            ax_energy.set_xlabel("Time")
            ax_energy.legend()
        
        def update(frame):
            with self.lock:
                pos = self.shared_state["positions"].copy()
                t = self.shared_state["time"]
                if self.track_energy:
                    KE = self.shared_state["KE"]
                    U = self.shared_state["U"]
                    E = self.shared_state["E"]
            
            # Update positions
            scat.set_offsets(pos)
            ax_pos.set_title(f"t = {t:.2f}")
            
            # Update energies if enabled
            if self.track_energy:
                t_history.append(t)
                KE_history.append(KE)
                U_history.append(U)
                E_history.append(E)
                
                line_KE.set_data(t_history, KE_history)
                line_U.set_data(t_history, U_history)
                line_E.set_data(t_history, E_history)
                
                ymin = min(min(KE_history), min(U_history), min(E_history))
                ymax = max(max(KE_history), max(U_history), max(E_history))
                ax_energy.set_ylim(ymin*1.1, ymax*1.1)
                ax_energy.set_xlim(0, t)
                return scat, line_KE, line_U, line_E
            
            return scat,

        ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

            
"""
@positions.setter
    def positions(self, arr):
        for b, pos in zip(self.bodies, arr):
            b.position = pos
"""







