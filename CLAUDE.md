# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Build
The project is built using `nvcc` (NVIDIA CUDA Compiler) and requires the CUDA Toolkit, FreeGLUT, and OpenGL.
```powershell
nvcc main.cu -o defense -std=c++17 ^
    -I"C:\path\to\vcpkg\installed\x64-windows\include" ^
    -Xlinker /LIBPATH:"C:\path\to\vcpkg\installed\x64-windows\lib" ^
    freeglut.lib opengl32.lib glu32.lib ^
    -O3 -arch=sm_75
```
*Note: Replace include and lib paths with your actual vcpkg installation location.*

### Run
The resulting executable is `defense.exe`.

## Architecture Overview

The system is designed for massive parallelism, with most physics and AI logic running in CUDA kernels.

### 1. Simulation Core (`simulation/`)
- **Integration Hub**: `defense_grade_simulation.cuh` manages the coordination between dynamics, ML, and sensors.
- **Physics Engine**:
    - `sixdof_dynamics.cuh`: Implements full 6-DOF flight dynamics using quaternions, ECEF coordinates, and J2 gravity perturbation.
    - `physx_engine.cuh`: Provides a PhysX-style simulation including atmospheric models (ISA), drag/lift coefficients, and continuous collision detection.
- **AI & ML**:
    - `physnode_pinn.cuh`: Physics-Informed Neural ODEs for trajectory prediction with uncertainty quantification (MC Dropout).
    - `hierarchical_rl.cuh`: A two-tier HRL architecture consisting of a "Strategist" (PPO/MAML) for high-level command and a "Pilot" (SAC + MPC) for terminal guidance.
    - `curriculum_learning.cuh`: Manages phased training progression from basic PN to adversarial self-play.
- **Sensors**: `sensor_simulation.cuh` simulates radar returns using a point-scatterer model and micro-Doppler signatures for target discrimination.

### 2. Rendering (`render/`)
- `renderer.h`: Uses OpenGL to render the 3D environment, a 2D tactical minimap, and a Heads-Up Display (HUD).

### 3. Core & IO (`core/`, `io/`)
- `main.cu`: Entry point, manages the main loop, CUDA context, and user input.
- `core/config.h`: Central repository for all simulation constants, RL hyperparameters, and physical constants.
- `io/file_io.h`: Handles telemetry logging to CSV and model checkpointing (serialization of `RLAgent` and `TrainingMetrics`).

## Key Technical Details
- **Coordinate Systems**: Uses NED (North-East-Down) for local simulation and ECEF (Earth-Centered, Earth-Fixed) for long-range ballistic trajectories.
- **Memory Management**: Heavy use of `__constant__` memory for global parameters and pre-allocated GPU buffers to avoid runtime allocations.
- **Guidance Logic**: Blends Proportional Navigation (PN) with RL-driven a-priori intent and MPC-based constraint satisfaction.
