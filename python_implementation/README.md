# Missile Defense Simulation - Python Implementation

A PyTorch-based implementation of the missile defense simulation with reinforcement learning, mirroring the CUDA implementation.

## Features

- **Physics Engine**: PhysX-style physics simulation with atmospheric modeling, aerodynamic drag/lift, and ballistic trajectory computation
- **PINN (Physics-Informed Neural Networks)**: Trajectory prediction that incorporates physics constraints
- **Reinforcement Learning**: 
  - PPO (Proximal Policy Optimization)
  - SAC (Soft Actor-Critic)
  - MAML (Model-Agnostic Meta-Learning) support
- **GPU Acceleration**: Full PyTorch CUDA support for parallel computation
- **Comprehensive Metrics**: Tracking of interception success, rewards, prediction accuracy, and more

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
python_implementation/
├── config.py              # Configuration constants and hyperparameters
├── physics_engine.py      # PhysX-style physics simulation
├── neural_networks.py     # Neural network architectures (PINN, PPO, SAC)
├── missile_simulation.py  # Main simulation logic
├── rl_agent.py           # RL agent implementation
├── main.py               # Training and evaluation script
├── visualization.py      # Plotting and visualization utilities
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Training

```bash
# Basic training with PPO
python main.py --mode train --model my_model --episodes 10000

# Train with SAC algorithm
python main.py --mode train --model my_model --algorithm sac --episodes 10000

# Resume training from checkpoint
python main.py --mode train --model my_model --resume

# Train with both PPO and SAC
python main.py --mode train --model my_model --algorithm both --episodes 10000
```

### Evaluation

```bash
# Evaluate latest checkpoint
python main.py --mode evaluate --model my_model

# Evaluate specific checkpoint
python main.py --mode evaluate --model my_model --checkpoint checkpoints/my_model_ep1000.checkpoint
```

### Visualization

```bash
# Plot training history
python visualization.py --model my_model --plot-history

# Generate training report
python visualization.py --model my_model --report
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RADAR_RANGE` | 8000.0 | Detection range (meters) |
| `COLLISION_RADIUS` | 100.0 | Interception radius (meters) |
| `MAX_MISSILES` | 50000 | Maximum active missiles |
| `NEURAL_HIDDEN` | 128 | Hidden layer size |
| `PPO_CLIP_EPSILON` | 0.2 | PPO clipping parameter |
| `INSIDE_RADAR_RATIO` | 0.8 | Training distribution (80% inside radar) |

## Reward Structure

- **Successful Interception**: +100
- **Fast Intercept Bonus**: Up to +50 (if < 5 seconds)
- **Early Detection Bonus**: Up to +20 (if < 2 seconds delay)
- **Accurate Prediction Bonus**: Up to +30 (if error < 500m)
- **Miss Penalty**: -80
- **Territory Hit Penalty**: -200
- **Retry Penalty**: -25 per additional interceptor

## Key Components

### Physics Engine
- International Standard Atmosphere (ISA) model
- Mach-dependent drag coefficients
- Lift and Magnus effect modeling
- Semi-implicit Euler integration with substeps

### PINN (Physics-Informed Neural Networks)
- Combines neural network predictions with physics constraints
- Outputs position correction, velocity prediction, and uncertainty
- Physics residual loss ensures F=ma constraint

### RL Algorithms

**PPO (Proximal Policy Optimization)**:
- Actor-Critic architecture with shared layers
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)

**SAC (Soft Actor-Critic)**:
- Twin Q-networks to reduce overestimation
- Automatic temperature adjustment
- Maximum entropy framework

## Comparison with CUDA Implementation

| Feature | CUDA | Python |
|---------|------|--------|
| Physics Computation | GPU Kernels | PyTorch Tensors |
| Neural Networks | Custom CUDA | PyTorch nn.Module |
| Parallelism | CUDA Blocks/Threads | PyTorch Batching |
| Memory Management | cudaMalloc | PyTorch Tensors |
| Visualization | OpenGL | Matplotlib |

## Performance Tips

1. **Use CUDA**: Ensure PyTorch is installed with CUDA support
2. **Batch Size**: Adjust `PPO_BATCH_SIZE` based on GPU memory
3. **Training Speed**: Increase `TRAINING_SPEED` for faster but less accurate physics
4. **Checkpoint Frequency**: Adjust `CHECKPOINT_INTERVAL` as needed

## License

Same as the parent CUDA project.

## References

- Original CUDA implementation in parent directory
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [PINN Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
