#ifndef CONFIG_H
#define CONFIG_H

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <GL/glut.h>
#include <algorithm>
#include <cmath>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <direct.h>
#include <filesystem>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

// ============================================================================
// CONSTANTS
// ============================================================================
#define MAX_MISSILES 50000
#define MAX_INTERCEPTORS 10000
#define BLOCK_SIZE 256
#define WORLD_SIZE 20000.0f
#define GROUND_LEVEL 0.0f
#define GRAVITY 9.81f
#define AIR_DENSITY 1.225f
#define DRAG_COEFFICIENT 0.3f
#define MISSILE_MASS 500.0f
#define INTERCEPTOR_MASS 200.0f
#define THRUST_FORCE 50000.0f
#define INTERCEPTOR_THRUST 80000.0f
#define MAX_TURN_RATE 0.15f
#define COLLISION_RADIUS 100.0f
#define RADAR_RANGE 8000.0f
#define NEURAL_HIDDEN 128
#define PREDICTION_STEPS 100
#define TRAINING_SPEED 100.0f
#define CHECKPOINT_INTERVAL 100
#define MAX_HISTORY 1000

// ============================================================================
// PHYSX SIMULATION CONSTANTS
// ============================================================================
#define PHYSX_SUBSTEPS 4
#define PHYSX_TIME_STEP 0.004f
#define PHYSX_RESTITUTION 0.3f
#define PHYSX_STATIC_FRICTION 0.5f
#define PHYSX_DYNAMIC_FRICTION 0.4f
#define PHYSX_LINEAR_DAMPING 0.01f
#define PHYSX_ANGULAR_DAMPING 0.05f

// Aerodynamic constants for PhysX simulation
#define REFERENCE_AREA_MISSILE 0.5f
#define REFERENCE_AREA_INTERCEPTOR 0.3f
#define LIFT_COEFFICIENT 0.4f
#define MAGNUS_COEFFICIENT 0.1f
#define SPIN_RATE 10.0f

// Atmospheric model
#define SEA_LEVEL_PRESSURE 101325.0f
#define SEA_LEVEL_TEMP 288.15f
#define TEMP_LAPSE_RATE 0.0065f
#define MOLAR_MASS_AIR 0.0289644f
#define GAS_CONSTANT 8.31447f

// ============================================================================
// RL ALGORITHM CONSTANTS
// ============================================================================
#define PPO_CLIP_EPSILON 0.2f
#define PPO_ENTROPY_COEF 0.01f
#define PPO_VALUE_COEF 0.5f
#define PPO_MAX_GRAD_NORM 0.5f
#define PPO_GAE_LAMBDA 0.95f
#define PPO_EPOCHS 10
#define PPO_BATCH_SIZE 64

#define SAC_TAU 0.005f
#define SAC_ALPHA 0.2f
#define SAC_TARGET_UPDATE_INTERVAL 2
#define SAC_BUFFER_SIZE 100000

#define MAML_INNER_LR 0.01f
#define MAML_OUTER_LR 0.001f
#define MAML_INNER_STEPS 5
#define MAML_META_BATCH_SIZE 4

// ============================================================================
// TRAINING DISTRIBUTION
// ============================================================================
#define INSIDE_RADAR_RATIO 0.80f
#define OUTSIDE_RADAR_RATIO 0.20f

// ============================================================================
// REWARD CONSTANTS
// ============================================================================
#define REWARD_INTERCEPT_SUCCESS 100.0f
#define REWARD_FAST_INTERCEPT_BONUS 50.0f
#define REWARD_EARLY_DETECTION 20.0f
#define REWARD_ACCURATE_PREDICTION 30.0f
#define PENALTY_MISS -80.0f
#define PENALTY_LATE_RESPONSE -20.0f
#define PENALTY_FUEL_WASTE -5.0f
#define PENALTY_COLLISION_MISS -100.0f

// Missile types
enum MissileType { ENEMY_MISSILE = 0, INTERCEPTOR_MISSILE = 1 };

// ============================================================================
// PHYSX-COMPATIBLE MISSILE STATE
// ============================================================================
struct PhysXState {
  float3 position;
  float3 velocity;
  float3 angularVelocity;
  float3 orientation;  // Euler angles (pitch, yaw, roll)
  float3 force;
  float3 torque;
  float mass;
  float inertia;
  float dragCoefficient;
  float liftCoefficient;
  float referenceArea;
};

// Missile state with full PhysX integration and logging
struct Missile {
  // PhysX state
  float3 position;
  float3 velocity;
  float3 acceleration;
  float3 angularVelocity;
  float3 orientation;
  float3 force;
  float3 torque;
  
  // Target and trajectory
  float3 target;
  float3 launchPos;
  float3 initialVelocity;
  float3 predictedImpact;
  
  // Physical properties
  float fuel;
  float mass;
  float dragCoefficient;
  float liftCoefficient;
  float referenceArea;
  float machNumber;
  float dynamicPressure;
  float altitude;
  float airDensity;
  float temperature;
  
  // State flags
  int active;
  int hit;
  int type;
  int insideRadar;
  int targetInsideRadar;
  
  // Timing
  float lifetime;
  float launchTime;
  float detectionTime;
  float interceptTime;
  
  // IDs and tracking
  int targetMissileId;
  int missileId;
  
  // Statistics
  float maxAltitude;
  float maxVelocity;
  float distanceTraveled;
  float fuelConsumed;
  float averageThrust;
  float maxAcceleration;
  float minDistanceToTarget;
  float predictionError;
  
  // PhysX substep data
  float3 prevPosition;
  float3 prevVelocity;
  float substepAccumulator;
};

// ============================================================================
// COMPREHENSIVE MISSILE EVENT LOG
// ============================================================================
struct MissileEvent {
  // Identification
  int missileId;
  int type;
  int episodeId;
  
  // Launch data
  float3 launchPos;
  float3 target;
  float3 initialVelocity;
  float launchTime;
  float launchAngle;
  float launchAzimuth;
  
  // Impact data
  float impactTime;
  float3 impactPos;
  float3 impactVelocity;
  float impactAngle;
  
  // Interception data
  int intercepted;
  int interceptorId;
  float responseTime;
  float detectionTime;
  float3 interceptionPoint;
  float interceptionDistance;
  
  // Trajectory statistics
  float maxAltitude;
  float maxVelocity;
  float totalDistance;
  float flightDuration;
  float fuelConsumed;
  float averageSpeed;
  float averageAcceleration;
  float maxAcceleration;
  
  // PhysX data
  float averageDragForce;
  float averageLiftForce;
  float averageMachNumber;
  float maxMachNumber;
  float averageDynamicPressure;
  
  // PINN prediction metrics
  float predictionError;
  float trajectoryDeviation;
  float pinnConfidence;
  
  // Radar data
  int targetInsideRadar;
  float distanceFromDefense;
  float radarCrossSection;
  
  // Reward calculation
  float reward;
  float rewardBreakdown[8];  // Individual reward components
  int rewardCalculated;      // Only if inside radar
};

// ============================================================================
// NEURAL NETWORK STRUCTURES
// ============================================================================
struct NeuralNetwork {
  float weights1[6 * NEURAL_HIDDEN];
  float bias1[NEURAL_HIDDEN];
  float weights2[NEURAL_HIDDEN * NEURAL_HIDDEN];
  float bias2[NEURAL_HIDDEN];
  float weights3[NEURAL_HIDDEN * 3];
  float bias3[3];
};

// Extended network for PPO Actor-Critic
struct ActorCriticNetwork {
  // Shared layers
  float sharedWeights1[12 * NEURAL_HIDDEN];  // Extended input for more features
  float sharedBias1[NEURAL_HIDDEN];
  float sharedWeights2[NEURAL_HIDDEN * NEURAL_HIDDEN];
  float sharedBias2[NEURAL_HIDDEN];
  
  // Actor head (policy)
  float actorWeights[NEURAL_HIDDEN * 6];  // 6 action dimensions
  float actorBias[6];
  float actorLogStd[6];  // Log standard deviation for continuous actions
  
  // Critic head (value)
  float criticWeights[NEURAL_HIDDEN * 1];
  float criticBias[1];
};

// SAC Networks
struct SACNetworks {
  ActorCriticNetwork policy;
  NeuralNetwork qNetwork1;
  NeuralNetwork qNetwork2;
  NeuralNetwork qTarget1;
  NeuralNetwork qTarget2;
  float logAlpha;
  float targetEntropy;
};

// MAML Meta-Learning State
struct MAMLState {
  // Fast weights (task-specific)
  float fastWeights1[6 * NEURAL_HIDDEN];
  float fastBias1[NEURAL_HIDDEN];
  float fastWeights2[NEURAL_HIDDEN * NEURAL_HIDDEN];
  float fastBias2[NEURAL_HIDDEN];
  float fastWeights3[NEURAL_HIDDEN * 3];
  float fastBias3[3];
  
  // Meta-gradients
  float metaGrad1[6 * NEURAL_HIDDEN];
  float metaGrad2[NEURAL_HIDDEN * NEURAL_HIDDEN];
  float metaGrad3[NEURAL_HIDDEN * 3];
  
  // Task buffers
  float taskRewards[MAML_META_BATCH_SIZE];
  int taskSteps[MAML_META_BATCH_SIZE];
  int currentTask;
  int innerStep;
};

// Experience replay buffer for SAC
struct ExperienceBuffer {
  float states[SAC_BUFFER_SIZE * 12];       // State vectors
  float actions[SAC_BUFFER_SIZE * 6];        // Action vectors
  float rewards[SAC_BUFFER_SIZE];            // Rewards
  float nextStates[SAC_BUFFER_SIZE * 12];    // Next state vectors
  int dones[SAC_BUFFER_SIZE];                // Episode termination flags
  int head;
  int size;
  int capacity;
};

// PPO Trajectory buffer
struct PPOBuffer {
  float states[PPO_BATCH_SIZE * 12];
  float actions[PPO_BATCH_SIZE * 6];
  float rewards[PPO_BATCH_SIZE];
  float values[PPO_BATCH_SIZE];
  float logProbs[PPO_BATCH_SIZE];
  float advantages[PPO_BATCH_SIZE];
  float returns[PPO_BATCH_SIZE];
  int count;
};

// ============================================================================
// RL AGENT WITH ADVANCED ALGORITHMS
// ============================================================================
struct RLAgent {
  // Core networks
  NeuralNetwork policyNet;
  NeuralNetwork valueNet;
  NeuralNetwork predictionNet;  // PINN for trajectory prediction
  
  // Advanced RL networks
  ActorCriticNetwork actorCritic;  // For PPO
  SACNetworks sacNets;             // For SAC
  MAMLState mamlState;             // For meta-learning
  
  // Experience buffers
  PPOBuffer ppoBuffer;
  ExperienceBuffer expBuffer;
  
  // Hyperparameters
  float learningRate;
  float epsilon;
  float gamma;
  float tau;           // Soft update coefficient
  float entropyCoef;   // Entropy regularization
  float valueCoef;     // Value loss coefficient
  float clipEpsilon;   // PPO clipping
  float gaeLambda;     // GAE lambda
  
  // Training state
  int episodeCount;
  float totalReward;
  float avgResponseTime;
  float bestSuccessRate;
  int updateStep;
  int trainingPhase;   // 0=warmup, 1=PPO, 2=SAC, 3=meta-learning
  
  // Statistics
  float recentRewards[100];
  float recentSuccessRates[100];
  int recentIndex;
  float movingAvgReward;
  float movingAvgSuccessRate;
};

// ============================================================================
// TRAINING METRICS (COMPREHENSIVE)
// ============================================================================
struct EpisodeMetrics {
  int episodeNum;
  int interceptSuccess;
  int interceptFail;
  int totalLaunched;
  int missilesInsideRadar;
  int missilesOutsideRadar;
  float successRate;
  float avgResponseTime;
  float avgInterceptDistance;
  float episodeReward;
  float epsilon;
  float learningRate;
  double timestamp;
  
  // Advanced metrics
  float avgPredictionError;
  float avgMachNumber;
  float avgFuelEfficiency;
  float avgTrajectoryDeviation;
  float ppoLoss;
  float valueLoss;
  float entropyLoss;
  float mamlMetaLoss;
};

// Training history for graphing
struct TrainingHistory {
  float successRates[MAX_HISTORY];
  float avgResponseTimes[MAX_HISTORY];
  float rewards[MAX_HISTORY];
  float predictionErrors[MAX_HISTORY];
  float ppoLosses[MAX_HISTORY];
  float valueLosses[MAX_HISTORY];
  int count;
};

// Global training metrics
struct TrainingMetrics {
  int interceptSuccess;
  int interceptFail;
  int totalLaunched;
  int insideRadarCount;
  int outsideRadarCount;
  float avgResponseTime;
  float avgInterceptDistance;
  float episodeReward;
  int totalMissilesFired;
  int totalInterceptorsFired;
  
  // PhysX metrics
  float avgDragForce;
  float avgLiftForce;
  float avgMachNumber;
  float maxMachNumber;
  
  // PINN metrics
  float avgPredictionError;
  float avgTrajectoryDeviation;
  float pinnAccuracy;
  
  // RL metrics
  float ppoLoss;
  float valueLoss;
  float entropyLoss;
  float qLoss;
  float mamlMetaLoss;
  
  // Reward components
  float totalInterceptReward;
  float totalSpeedBonus;
  float totalPredictionBonus;
  float totalPenalties;
};

#endif // CONFIG_H
