# AI Missile Defense Simulation

A CUDA + OpenGL missile defense simulator with PhysX-style flight dynamics, Physics-Informed Neural Networks (PINN), and advanced RL structures (PPO/SAC/MAML). The simulator supports mouse click targeting, radar-gated rewards, CSV logging, and real-time visualization with a minimap.

## Highlights
- PhysX-style physics (drag, lift, substep integration, collision checks)
- PINN trajectory prediction with physics constraints and uncertainty
- RL scaffolding: PPO/SAC/MAML, GAE, and experience buffers
- Inverse ballistic trajectory solver for enemy missile launches
- Two-layer guidance for interceptors (RL high-level + MPC refinement)
- 80/20 training distribution (inside/outside radar)
- Extensive CSV logging (60+ parameters)
- Real-time HUD + minimap with target marker

## Requirements
- Windows
- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc)
- FreeGLUT + OpenGL (vcpkg recommended)

## Build
From the project root:

```bash
nvcc main.cu -o defense -std=c++17 -I"E:\karano-kokoro\vcpkg\installed\x64-windows\include" -Xlinker /LIBPATH:"E:\karano-kokoro\vcpkg\installed\x64-windows\lib" freeglut.lib opengl32.lib glu32.lib -O3 -arch=sm_75
```

If you use a different vcpkg path or GPU architecture, update the include/library paths and the `-arch` flag accordingly.

## Run
```bash
./defense.exe
```
You will be prompted to start a new model or continue from an existing checkpoint.

## Controls
- **Left Click (3D)**: Set target and launch enemy missile
- **Left Click (Minimap)**: Launch enemy missile at map position
- **Right Drag**: Rotate camera
- **T**: Toggle training mode (1x/100x speed)
- **S**: Save checkpoint manually
- **SPACE**: Launch 10 test missiles (80/20 distribution)
- **M**: Toggle minimap
- **R**: Reset simulation
- **ESC**: Save and exit

## Project Structure
- [main.cu](main.cu): App entry point, simulation loop, UI, input handling
- [core/config.h](core/config.h): Constants, structs, and model definitions
- [core/globals.h](core/globals.h): Global variables and shared state
- [simulation/physx_engine.cuh](simulation/physx_engine.cuh): Physics engine + inverse solver + MPC
- [simulation/kernels.cuh](simulation/kernels.cuh): CUDA kernels for simulation and RL
- [simulation/neural_net.cuh](simulation/neural_net.cuh): PINN and RL networks
- [render/renderer.h](render/renderer.h): OpenGL rendering and HUD/minimap
- [io/file_io.h](io/file_io.h): CSV and training logs

## Logging
- **Missile CSV**: `logs/<model>_missiles.csv`
- **Training Log**: `logs/<model>_training.log`
- **Checkpoints**: `checkpoints/<model>_*.checkpoint`

## Notes
- Enemy missiles are launched using the inverse ballistic solver and simulated as ballistic (no thrust) to align with target markers.
- Training rewards are only computed for targets inside the radar range.

## Troubleshooting
- **Missing GLUT headers**: Ensure FreeGLUT is installed and include paths are correct.
- **Linker errors**: Verify `opengl32.lib`, `glu32.lib`, and `freeglut.lib` paths.
- **Black screen**: Update GPU drivers and confirm OpenGL context creation.

## License
This project is provided as-is for research and simulation purposes.
