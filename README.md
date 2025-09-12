# PyPebble: An N-Body Simulation Toolkit

PyPebble is a Python model to handle N-body problems, specifically geared towards gravitational simulations (with more features planned for the future). This code is still under development.
Currently, inputs and outputs are configured to use the following units:
- Distance: parsecs
- Mass: solar masses
- Time: megayears

----------------------------------------

## Installation

Using pip:
```bash
pip install pyPebble
```
Clone this repository and then install manually with:
```bash
pip install -e .
```
------------------------------------------------------

## Usage

### Creating a System
You can define the properties of particles directly by passing arrays of positions, velocities, and masses:

from pypebble import Pebbles
```python
system = Pebbles(positions, velocities, masses)
```
Alternatively, you can create a disc of particles using the built-in generator:
```python
system = Pebbles().create_disc(
    n_particles=100,
    r_max=10,
    center=[0, 0],
    ang_vel=0.05,
    v_sigma=None,
    total_mass=None,
    particle_mass=None,
    distribution="uniform"
)
```
### Setting up the Simulation
With a system defined, initialize the simulation using setup:
```python
simulation = system.setup(
    softening=1e-2,
    bounds=20,
    smooth_len=10,
    t_start=0,
    t_finish=50,
    nsteps=1000,
    Enable_GPU=True,
    save_output=None
)
```
### Live Animation
```python
simulation.start_animation()
```

### Running the Simulation
Run with:
```python
simulation.run()
```
The simulation currently saves output data (time, position, velocity, and mass) to an .h5 file.
Future improvements will include better ways to run and visualize simulations.

--------------------------------------------------------------

## To do
- Support for multiple unit systems
- Additional initial condition generators
- General Optimisations
- GUI 
- Energy conservation checks
- Multiprocessing option for simulation
- Multithreading for writing to file
- Allow simulations beyond a 2D plane
- Realistic velocity and density distributions

-------------------------------------------------------------

## License
MIT License.

