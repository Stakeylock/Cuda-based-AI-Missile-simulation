# Cuda-based AI Missile Defense Simulation

A high-fidelity, GPU-accelerated missile defense simulator enabling research into advanced intercept algorithms, Physics-Informed Neural Networks (PINNs), and Hierarchical Reinforcement Learning (HRL). 

This project simulates complex engagement scenarios involving maneuvering hypersonic threats (HGVs), authentic sensor phenomenology (RCS/Micro-Doppler), and multi-tiered defense strategies running directly on CUDA for massive parallelism.

## 🚀 Key Features

*   **Defense-Grade Physics**: 
    *   Full **6-DOF flight dynamics** with quaternion attitude control.
    *   **WGS-84 ECEF** coordinate system for realistic long-range trajectories.
    *   **J2 Perturbation** gravity models.
    *   Variable mass/inertia modeling (fuel burn) and Mach-dependent aerodynamics.
*   **Advanced AI & ML**:
    *   **PhysNODE (Physics-Informed Neural ODE)**: Differentiable physics solver for trajectory prediction with uncertainty quantification.
    *   **Hierarchical RL (HRL)**: A two-tier agent architecture featuring a "Strategist" (Meta-Controller) for resource allocation and a "Pilot" (SAC+MPC) for terminal guidance.
    *   **Curriculum Learning**: Auto-phased training from basic navigation to adversarial self-play.
*   **High-Fidelity Sensor Simulation**:
    *   **RCS & Micro-Doppler**: Real-time GPU generation of radar signatures for warheads, decoys, and debris.
    *   **Discrimination**: Neural networks for target classification based on motion and spectral features.
*   **Real-Time Visualization**: OpenGL-based rendering with HUD, 3D tactical view, and tactical minimap.

---

## 📂 Module Breakdown

This project is organized into modular components, leveraging CUDA for heavy computational tasks.

### 1. Simulation Core (`simulation/`)

This is the heart of the engine, containing the CUDA kernels and physics logic.

*   **`defense_grade_simulation.cuh`**  
    *   **The Integration Hub**: Connects all subsystems (Dynamics, ML, Sensors) into a cohesive simulation loop.
    *   Manages GPU memory for extended missile states, neural networks, and sensor buffers.
    *   Handles the switching between "Legacy" (simple particle) and "Defense Grade" (6-DOF) simulation modes.

*   **`sixdof_dynamics.cuh`**  
    *   **Flight Dynamics Engine**: Implements the full equations of motion for a rigid body in 3D space.
    *   **Aerodynamics Database**: Looks up $C_L, C_D, C_m$ coefficients based on Mach number, Angle of Attack ($\alpha$), and Sideslip ($\beta$).
    *   Includes specific models for **Hypersonic Glide Vehicles (HGV)** (waverider aerodynamics).
    *   Handles coordinate conversions (NED $\leftrightarrow$ ECEF $\leftrightarrow$ Geodetic).
    *   **MCI**: Tracks Mass, Center of Gravity, and Inertia changes in real-time as fuel is consumed.

*   **`sensor_simulation.cuh`**  
    *   **"The Eyes"**: Simulates electromagnetic radar returns.
    *   **Point-Scatterer Model**: Reconstructs RCS from multiple scattering centers on a target.
    *   **Micro-Doppler**: Simulates the specific frequency modulations caused by target spin, precession, and wobble.
    *   **Discrimination**: Extracts features (bandwidth, centroid, periodicity) to distinguish heavy warheads from light decoys (balloons/cones).

*   **`physnode_pinn.cuh`**  
    *   **Prediction Engine**: A Physics-Informed Neural Network (PINN) that predicts where a target will be.
    *   **Hybrid Architecture**: Combines a 1D CNN for feature extraction with a differentiable physics solver.
    *   **Parameter Estimation**: Infers unknown physical properties (mass, drag coefficient, Lift-to-Drag ratio) from observation history.
    *   **Uncertainty**: Uses Monte Carlo Dropout to estimate confidence intervals for its predictions.

*   **`hierarchical_rl.cuh`**  
    *   **The Brains**: Implements the nested control architecture.
    *   **Strategist (Level 1)**: A low-frequency PPO/MAML agent that looks at the global picture (resources, threat levels) to assign targets and issue commands.
    *   **Pilot (Level 2)**: A high-frequency SAC agent paired with Model Predictive Control (MPC) to fly the interceptor, satisfy g-load constraints, and achieve hit-to-kill.

*   **`curriculum_learning.cuh`**  
    *   **The Teacher**: Manages the training progression.
    *   **Phases**:
        1.  **Basic PN**: Non-maneuvering targets to learn fundamental guidance.
        2.  **Maneuvering**: Targets that weave and roll.
        3.  **Multi-Agent**: Saturation attacks with decoys.
        4.  **Adversarial**: Self-play against an adaptive "Threat Agent" that learns to evade.
    *   **Auto-Advancement**: Monitors success rates to automatically promote the agent to the next difficulty tier.

*   **`cuda_utils.cuh`**  
    *   Helper math functions (`float3` operators, dot/cross products) and CUDA error checking macros.

### 2. Rendering (`render/`)

*   **`renderer.h`**  
    *   OpenGL visualization engine.
    *   Draws the 3D world (missiles, terrain, trajectories).
    *   Renders the 2D HUD (Heads-Up Display) and tactical minimap.
    *   Visualizes uncertainty ellipsoids and predicted trajectories.

### 3. Core & IO (`core/`, `io/`)

*   **`main.cu`**: The application entry point. Sets up the CUDA context, initializes OpenGL, and runs the main loop.
*   **`core/config.h`**: Global constants, simulation parameters, and physics settings.
*   **`io/file_io.h`**: Handles CSV logging of telemetry data (position, velocity, events) for post-analysis.

---

## 🛠 Prerequisites

*   **OS**: Windows 10/11
*   **GPU**: NVIDIA GPU with CUDA Compute Capability 7.5+ (RTX 20-series or newer recommended for full fidelity).
*   **Compiler**: MSVC (Visual Studio) with C++17 support.
*   **Libraries**:
    *   **CUDA Toolkit** (11.0+)
    *   **FreeGLUT** (for windowing)
    *   **GLEW/OpenGL** (for rendering)

*Recommended*: Use `vcpkg` to install OpenGL/GLUT dependencies.

---

## 🔨 Build Instructions

Run the following command from the project root (ensure `nvcc` is in your PATH). 

**Note**: You may need to adjust the include/lib paths to match your `vcpkg` installation location.

```powershell
nvcc main.cu -o defense -std=c++17 ^
    -I"C:\path\to\vcpkg\installed\x64-windows\include" ^
    -Xlinker /LIBPATH:"C:\path\to\vcpkg\installed\x64-windows\lib" ^
    freeglut.lib opengl32.lib glu32.lib ^
    -O3 -arch=sm_75
```

---

## 🎮 Controls

| Key / Action | Function |
| :--- | :--- |
| **Left Click (3D)** | Set target & launch enemy missile (Manual Mode) |
| **Left Click (Map)** | Quick launch enemy at map coordinates |
| **Right Drag** | Rotate Camera |
| **Scroll** | Zoom In/Out |
| **SPACE** | Launch Test Wave (80/20 distribution) |
| **T** | Toggle Training Mode (Speed up 100x) |
| **S** | Save Checkpoint |
| **M** | Toggle Minimap |
| **R** | Reset Simulation |
| **ESC** | Exit & Save |

---

## 📊 Logging

The simulation automatically logs data to the `logs/` directory:

*   **`*_missiles.csv`**: Detailed telemetry for every missile (Pos, Vel, Accel, Mass, Status).
*   **`*_training.log`**: RL training metrics (Reward, Loss, Success Rate).

---

## 🔬 Research Context

This project implements concepts from:
*   *Physics-Informed Neural Networks (PINNs)* for trajectory estimation.
*   *Hierarchical Reinforcement Learning* for multi-agent coordination.
*   *Micro-Doppler Signature Analysis* for target discrimination.
