#ifndef HIERARCHICAL_RL_CUH
#define HIERARCHICAL_RL_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"
#include "physnode_pinn.cuh"

// ============================================================================
// HIERARCHICAL REINFORCEMENT LEARNING (HRL) ARCHITECTURE
// Based on research document: "Intelligent Guidance & Control: RL and Hybrid Architectures"
//
// Level 1: The Strategist (Meta-Controller) - PPO/MAML
//   - Frequency: 1-10 Hz
//   - Input: Global state (threats, predicted impacts, resources)
//   - Output: High-level commands (target selection, engagement geometry)
//
// Level 2: The Pilot (Guidance Controller) - SAC + MPC
//   - Frequency: 100-1000 Hz  
//   - Input: Relative geometry, LOS rates, constraints
//   - Output: Continuous fin commands
// ============================================================================

// ============================================================================
// STRATEGIST STATE AND ACTION SPACES
// ============================================================================

#define STRATEGIST_STATE_DIM 64     // Global situation awareness
#define STRATEGIST_ACTION_DIM 8     // High-level command vector
#define MAX_TRACKED_THREATS 20      // Maximum threats to track
#define MAX_INTERCEPTORS_AVAILABLE 10

// Global threat assessment
struct ThreatInfo {
    int threatId;
    int targetType;                 // From TargetType enum
    float3 position;
    float3 velocity;
    float3 predictedImpact;
    float timeToImpact;
    float threatPriority;           // 0-1, higher = more dangerous
    float discriminationConfidence; // Confidence it's a real warhead
    float interceptProbability;     // P(kill) if engaged
    bool isEngaged;                 // Already has interceptor assigned
    int assignedInterceptorId;
};

// Global defense state
struct DefenseState {
    ThreatInfo threats[MAX_TRACKED_THREATS];
    int numThreats;
    
    int availableInterceptors;
    int interceptorsInFlight;
    int successfulIntercepts;
    int missedThreats;
    
    float radarCoverage;           // Fraction of threats being tracked
    float avgDiscriminationConf;   // Average confidence in threat classification
    float timeUntilSaturation;     // Time until defense is overwhelmed
    
    // Resource status
    float totalFuelRemaining;
    float avgInterceptorHealth;
};

// Strategist high-level commands
enum StrategistCommand {
    CMD_ENGAGE_HIGHEST_PRIORITY = 0,
    CMD_ENGAGE_CLOSEST = 1,
    CMD_ENGAGE_SPECIFIC_TARGET = 2,
    CMD_HOLD_FIRE = 3,              // Wait for better discrimination
    CMD_SALVO_FIRE = 4,             // Multiple interceptors on one target
    CMD_PREFERENTIAL_DEFENSE = 5,   // Defend specific area
    CMD_SHOOT_LOOK_SHOOT = 6,       // Fire, wait for result, fire again
    CMD_SHOOT_SHOOT_LOOK = 7        // Double-tap immediately
};

// Strategist action output
struct StrategistAction {
    StrategistCommand command;
    int targetThreatId;            // Which threat to engage (-1 for auto)
    float engagementUrgency;       // 0-1, affects pilot aggressiveness
    float3 preferredInterceptPoint; // Suggested intercept geometry
    float confidenceThreshold;     // Min discrimination confidence to engage
    int numInterceptors;           // For salvo tactics
    float waitTime;                // For shoot-look-shoot timing
};

// ============================================================================
// PILOT STATE AND ACTION SPACES
// ============================================================================

#define PILOT_STATE_DIM 24          // Relative engagement geometry
#define PILOT_ACTION_DIM 4          // Fin deflections + throttle

// Relative engagement state (for single interceptor-target pair)
struct EngagementState {
    // Relative geometry
    float range;                   // Distance to target
    float rangeRate;               // Closing velocity
    float3 lineOfSight;            // Unit vector to target
    float3 lineOfSightRate;        // LOS rotation rate (critical for PN)
    
    // Target motion
    float3 targetVelocity;
    float3 targetAcceleration;     // Estimated from tracking
    float targetMachNumber;
    
    // Interceptor state
    float3 interceptorVelocity;
    float fuel;
    float machNumber;
    float dynamicPressure;
    float angleOfAttack;
    
    // Predicted geometry
    float zeroEffortMiss;          // ZEM - key metric
    float timeToGo;                // Estimated time to intercept
    float3 predictedInterceptPoint;
    
    // Constraints
    float maxAcceleration;         // Current g-limit
    float maxTurnRate;             // Structural limit
    bool inTerminalPhase;          // Close enough for terminal guidance
};

// Pilot control output
struct PilotAction {
    float pitchCommand;            // -1 to 1 (normalized fin deflection)
    float yawCommand;              // -1 to 1
    float rollCommand;             // -1 to 1
    float throttleCommand;         // 0 to 1
    
    // Derived quantities for MPC
    float3 commandedAcceleration;
    float3 commandedDirection;
};

// ============================================================================
// STRATEGIST NETWORK (PPO-based)
// ============================================================================

struct StrategistNetwork {
    // Shared feature extractor
    float sharedWeights1[STRATEGIST_STATE_DIM * 256];
    float sharedBias1[256];
    float sharedWeights2[256 * 128];
    float sharedBias2[128];
    
    // Actor head (policy)
    float actorWeights[128 * STRATEGIST_ACTION_DIM];
    float actorBias[STRATEGIST_ACTION_DIM];
    float actorLogStd[STRATEGIST_ACTION_DIM];
    
    // Critic head (value)
    float criticWeights[128];
    float criticBias;
    
    // Attention mechanism for threat prioritization
    float attentionQuery[128 * 32];
    float attentionKey[128 * 32];
    float attentionValue[128 * 32];
};

// Encode global defense state to network input
__device__ inline void encodeDefenseState(
    DefenseState* state,
    float* stateVector  // Output: [STRATEGIST_STATE_DIM]
) {
    int idx = 0;
    
    // Global metrics
    stateVector[idx++] = (float)state->numThreats / MAX_TRACKED_THREATS;
    stateVector[idx++] = (float)state->availableInterceptors / MAX_INTERCEPTORS_AVAILABLE;
    stateVector[idx++] = (float)state->interceptorsInFlight / MAX_INTERCEPTORS_AVAILABLE;
    stateVector[idx++] = state->radarCoverage;
    stateVector[idx++] = state->avgDiscriminationConf;
    stateVector[idx++] = fminf(state->timeUntilSaturation / 60.0f, 1.0f);  // Normalize to 60s
    
    // Top threats summary (sorted by priority)
    for (int i = 0; i < 5 && i < state->numThreats; i++) {
        ThreatInfo* t = &state->threats[i];
        stateVector[idx++] = t->position.x / 20000.0f;
        stateVector[idx++] = t->position.y / 10000.0f;
        stateVector[idx++] = t->position.z / 20000.0f;
        stateVector[idx++] = t->threatPriority;
        stateVector[idx++] = t->discriminationConfidence;
        stateVector[idx++] = t->timeToImpact / 30.0f;  // Normalize to 30s
        stateVector[idx++] = t->interceptProbability;
        stateVector[idx++] = t->isEngaged ? 1.0f : 0.0f;
    }
    
    // Pad remaining
    while (idx < STRATEGIST_STATE_DIM) {
        stateVector[idx++] = 0.0f;
    }
}

// Strategist forward pass
__device__ inline void strategistForward(
    StrategistNetwork* net,
    float* stateVector,
    float* actionMean,    // Output: [STRATEGIST_ACTION_DIM]
    float* actionStd,     // Output: [STRATEGIST_ACTION_DIM]
    float* value          // Output: scalar
) {
    // Shared layers
    float hidden1[256];
    for (int i = 0; i < 256; i++) {
        float sum = net->sharedBias1[i];
        for (int j = 0; j < STRATEGIST_STATE_DIM; j++) {
            sum += stateVector[j] * net->sharedWeights1[j * 256 + i];
        }
        hidden1[i] = fmaxf(0.0f, sum);  // ReLU
    }
    
    float hidden2[128];
    for (int i = 0; i < 128; i++) {
        float sum = net->sharedBias2[i];
        for (int j = 0; j < 256; j++) {
            sum += hidden1[j] * net->sharedWeights2[j * 128 + i];
        }
        hidden2[i] = fmaxf(0.0f, sum);
    }
    
    // Actor head
    for (int i = 0; i < STRATEGIST_ACTION_DIM; i++) {
        float sum = net->actorBias[i];
        for (int j = 0; j < 128; j++) {
            sum += hidden2[j] * net->actorWeights[j * STRATEGIST_ACTION_DIM + i];
        }
        actionMean[i] = tanhf(sum);  // Bounded output
        actionStd[i] = expf(net->actorLogStd[i]);
        actionStd[i] = fmaxf(0.01f, fminf(1.0f, actionStd[i]));
    }
    
    // Critic head
    float valueSum = net->criticBias;
    for (int j = 0; j < 128; j++) {
        valueSum += hidden2[j] * net->criticWeights[j];
    }
    *value = valueSum;
}

// Decode action vector to strategic command
__device__ inline StrategistAction decodeStrategistAction(
    float* actionVector,  // [STRATEGIST_ACTION_DIM]
    DefenseState* state
) {
    StrategistAction action;
    
    // Command type from first dimension (discretized)
    float cmdFloat = (actionVector[0] + 1.0f) * 4.0f;  // Map [-1,1] to [0,8]
    action.command = (StrategistCommand)fminf(7.0f, fmaxf(0.0f, cmdFloat));
    
    // Target selection (index into threat list)
    float targetFloat = (actionVector[1] + 1.0f) * 0.5f * state->numThreats;
    action.targetThreatId = (int)fminf((float)(state->numThreats - 1), fmaxf(0.0f, targetFloat));
    
    // Urgency
    action.engagementUrgency = (actionVector[2] + 1.0f) * 0.5f;
    
    // Preferred intercept point direction
    action.preferredInterceptPoint = make_float3(
        actionVector[3] * 5000.0f,
        actionVector[4] * 5000.0f + 2000.0f,  // Bias upward
        actionVector[5] * 5000.0f
    );
    
    // Confidence threshold
    action.confidenceThreshold = (actionVector[6] + 1.0f) * 0.5f;  // 0 to 1
    
    // Number of interceptors
    action.numInterceptors = (int)fmaxf(1.0f, fminf(3.0f, (actionVector[7] + 1.0f) * 1.5f + 1.0f));
    
    return action;
}

// ============================================================================
// PILOT NETWORK (SAC-based with continuous actions)
// ============================================================================

struct PilotNetwork {
    // Policy network (actor)
    float policyWeights1[PILOT_STATE_DIM * 128];
    float policyBias1[128];
    float policyWeights2[128 * 64];
    float policyBias2[64];
    float policyMeanWeights[64 * PILOT_ACTION_DIM];
    float policyMeanBias[PILOT_ACTION_DIM];
    float policyLogStdWeights[64 * PILOT_ACTION_DIM];
    float policyLogStdBias[PILOT_ACTION_DIM];
    
    // Q-networks (twin critics)
    float q1Weights1[(PILOT_STATE_DIM + PILOT_ACTION_DIM) * 128];
    float q1Bias1[128];
    float q1Weights2[128 * 64];
    float q1Bias2[64];
    float q1WeightsOut[64];
    float q1BiasOut;
    
    float q2Weights1[(PILOT_STATE_DIM + PILOT_ACTION_DIM) * 128];
    float q2Bias1[128];
    float q2Weights2[128 * 64];
    float q2Bias2[64];
    float q2WeightsOut[64];
    float q2BiasOut;
    
    // Target networks (for stable learning)
    float q1TargetWeights1[(PILOT_STATE_DIM + PILOT_ACTION_DIM) * 128];
    float q1TargetBias1[128];
    float q1TargetWeights2[128 * 64];
    float q1TargetBias2[64];
    float q1TargetWeightsOut[64];
    float q1TargetBiasOut;
    
    float q2TargetWeights1[(PILOT_STATE_DIM + PILOT_ACTION_DIM) * 128];
    float q2TargetBias1[128];
    float q2TargetWeights2[128 * 64];
    float q2TargetBias2[64];
    float q2TargetWeightsOut[64];
    float q2TargetBiasOut;
    
    // Entropy temperature (auto-tuned)
    float logAlpha;
    float targetEntropy;
};

// Encode engagement state for pilot
__device__ inline void encodeEngagementState(
    EngagementState* state,
    float* stateVector  // Output: [PILOT_STATE_DIM]
) {
    int idx = 0;
    
    // Relative geometry (most critical)
    stateVector[idx++] = state->range / 10000.0f;
    stateVector[idx++] = state->rangeRate / 1000.0f;
    stateVector[idx++] = state->lineOfSight.x;
    stateVector[idx++] = state->lineOfSight.y;
    stateVector[idx++] = state->lineOfSight.z;
    stateVector[idx++] = state->lineOfSightRate.x * 10.0f;
    stateVector[idx++] = state->lineOfSightRate.y * 10.0f;
    stateVector[idx++] = state->lineOfSightRate.z * 10.0f;
    
    // Target motion
    stateVector[idx++] = state->targetVelocity.x / 1000.0f;
    stateVector[idx++] = state->targetVelocity.y / 1000.0f;
    stateVector[idx++] = state->targetVelocity.z / 1000.0f;
    stateVector[idx++] = state->targetMachNumber / 5.0f;
    
    // Interceptor state
    stateVector[idx++] = state->interceptorVelocity.x / 1000.0f;
    stateVector[idx++] = state->interceptorVelocity.y / 1000.0f;
    stateVector[idx++] = state->interceptorVelocity.z / 1000.0f;
    stateVector[idx++] = state->fuel / 15.0f;
    stateVector[idx++] = state->machNumber / 3.0f;
    stateVector[idx++] = state->angleOfAttack / (M_PI / 4.0f);
    
    // Key metrics
    stateVector[idx++] = state->zeroEffortMiss / 1000.0f;
    stateVector[idx++] = state->timeToGo / 20.0f;
    
    // Constraints
    stateVector[idx++] = state->maxAcceleration / 500.0f;
    stateVector[idx++] = state->inTerminalPhase ? 1.0f : 0.0f;
    
    // Pad
    while (idx < PILOT_STATE_DIM) {
        stateVector[idx++] = 0.0f;
    }
}

// Pilot policy forward pass
__device__ inline void pilotPolicyForward(
    PilotNetwork* net,
    float* stateVector,
    float* actionMean,   // Output: [PILOT_ACTION_DIM]
    float* actionLogStd, // Output: [PILOT_ACTION_DIM]
    curandState* randState,
    float* sampledAction // Output: [PILOT_ACTION_DIM]
) {
    // Hidden layer 1
    float hidden1[128];
    for (int i = 0; i < 128; i++) {
        float sum = net->policyBias1[i];
        for (int j = 0; j < PILOT_STATE_DIM; j++) {
            sum += stateVector[j] * net->policyWeights1[j * 128 + i];
        }
        hidden1[i] = fmaxf(0.0f, sum);
    }
    
    // Hidden layer 2
    float hidden2[64];
    for (int i = 0; i < 64; i++) {
        float sum = net->policyBias2[i];
        for (int j = 0; j < 128; j++) {
            sum += hidden1[j] * net->policyWeights2[j * 64 + i];
        }
        hidden2[i] = fmaxf(0.0f, sum);
    }
    
    // Mean and log std outputs
    for (int i = 0; i < PILOT_ACTION_DIM; i++) {
        float meanSum = net->policyMeanBias[i];
        float logStdSum = net->policyLogStdBias[i];
        for (int j = 0; j < 64; j++) {
            meanSum += hidden2[j] * net->policyMeanWeights[j * PILOT_ACTION_DIM + i];
            logStdSum += hidden2[j] * net->policyLogStdWeights[j * PILOT_ACTION_DIM + i];
        }
        actionMean[i] = tanhf(meanSum);
        actionLogStd[i] = fmaxf(-20.0f, fminf(2.0f, logStdSum));  // Clamp
    }
    
    // Sample action with reparameterization trick
    if (randState != nullptr) {
        for (int i = 0; i < PILOT_ACTION_DIM; i++) {
            float noise = curand_normal(randState);
            float std = expf(actionLogStd[i]);
            sampledAction[i] = tanhf(actionMean[i] + std * noise);
        }
    }
}

// Q-network forward pass
__device__ inline float pilotQForward(
    float* qWeights1, float* qBias1,
    float* qWeights2, float* qBias2,
    float* qWeightsOut, float qBiasOut,
    float* stateVector,
    float* actionVector
) {
    // Concatenate state and action
    float input[PILOT_STATE_DIM + PILOT_ACTION_DIM];
    for (int i = 0; i < PILOT_STATE_DIM; i++) input[i] = stateVector[i];
    for (int i = 0; i < PILOT_ACTION_DIM; i++) input[PILOT_STATE_DIM + i] = actionVector[i];
    
    // Hidden layer 1
    float hidden1[128];
    for (int i = 0; i < 128; i++) {
        float sum = qBias1[i];
        for (int j = 0; j < PILOT_STATE_DIM + PILOT_ACTION_DIM; j++) {
            sum += input[j] * qWeights1[j * 128 + i];
        }
        hidden1[i] = fmaxf(0.0f, sum);
    }
    
    // Hidden layer 2
    float hidden2[64];
    for (int i = 0; i < 64; i++) {
        float sum = qBias2[i];
        for (int j = 0; j < 128; j++) {
            sum += hidden1[j] * qWeights2[j * 64 + i];
        }
        hidden2[i] = fmaxf(0.0f, sum);
    }
    
    // Output
    float qValue = qBiasOut;
    for (int j = 0; j < 64; j++) {
        qValue += hidden2[j] * qWeightsOut[j];
    }
    
    return qValue;
}

// ============================================================================
// HYBRID RL-MPC GUIDANCE
// RL provides high-level intent, MPC ensures constraint satisfaction
// ============================================================================

// MPC constraints for interceptor
struct MPCConstraints {
    float maxNormalAccel;     // Max lateral acceleration (g's)
    float maxAxialAccel;      // Max along-velocity acceleration
    float maxAngleOfAttack;   // Stall limit
    float maxFinDeflection;   // Actuator limit
    float maxFinRate;         // Actuator rate limit
    float minAltitude;        // Terrain avoidance
};

// MPC solution
struct MPCSolution {
    float3 commandedAccel;    // Optimal acceleration command
    float throttle;           // Optimal throttle
    float cost;               // Objective value
    bool feasible;            // Whether constraints are satisfied
    int iterations;           // Solver iterations
};

// Solve MPC to refine RL action
__device__ inline MPCSolution solvePilotMPC(
    EngagementState* state,
    PilotAction* rlAction,        // Desired action from RL
    MPCConstraints* constraints,
    float dt
) {
    MPCSolution sol;
    memset(&sol, 0, sizeof(MPCSolution));
    sol.feasible = true;
    
    // RL provides desired direction
    float3 desiredAccel = rlAction->commandedAcceleration;
    float desiredThrottle = rlAction->throttleCommand;
    
    // Convert fin commands to acceleration commands
    float pitchAccel = rlAction->pitchCommand * constraints->maxNormalAccel * GRAVITY;
    float yawAccel = rlAction->yawCommand * constraints->maxNormalAccel * GRAVITY;
    
    desiredAccel = make_float3(
        yawAccel,
        pitchAccel,
        desiredThrottle * constraints->maxAxialAccel * GRAVITY
    );
    
    // Project onto constraint set
    float accelMag = length(desiredAccel);
    float maxAccelMag = constraints->maxNormalAccel * GRAVITY;
    
    if (accelMag > maxAccelMag) {
        desiredAccel = desiredAccel * (maxAccelMag / accelMag);
    }
    
    // Check angle of attack constraint
    float3 velDir = normalize(state->interceptorVelocity);
    float3 accelDir = length(desiredAccel) > 0.01f ? normalize(desiredAccel) : velDir;
    float predictedAOA = acosf(fmaxf(-1.0f, fminf(1.0f, dot(velDir, accelDir))));
    
    if (predictedAOA > constraints->maxAngleOfAttack) {
        // Reduce command to stay within AOA limit
        float scale = constraints->maxAngleOfAttack / predictedAOA;
        desiredAccel = desiredAccel * scale;
        sol.feasible = false;  // Had to modify command
    }
    
    // Check altitude constraint
    float predictedAlt = state->interceptorVelocity.y > 0 ? 
                         state->interceptorVelocity.y : 
                         state->interceptorVelocity.y + desiredAccel.y * dt;
    // (Simplified - full MPC would predict trajectory)
    
    // Throttle constraint
    sol.throttle = fmaxf(0.0f, fminf(1.0f, desiredThrottle));
    
    sol.commandedAccel = desiredAccel;
    sol.cost = length(desiredAccel - rlAction->commandedAcceleration);  // How much we modified
    sol.iterations = 1;  // Simplified direct projection
    
    return sol;
}

// ============================================================================
// COMBINED HIERARCHICAL CONTROL LAW
// ============================================================================

// Full HRL agent structure
struct HierarchicalRLAgent {
    StrategistNetwork strategist;
    PilotNetwork pilot;
    
    // Strategist state
    DefenseState defenseState;
    StrategistAction currentStrategy;
    float strategistUpdateTimer;
    float strategistUpdateInterval;  // Typically 0.1 to 1.0 seconds
    
    // Pilot state per interceptor
    EngagementState engagementStates[MAX_INTERCEPTORS_AVAILABLE];
    PilotAction pilotActions[MAX_INTERCEPTORS_AVAILABLE];
    MPCConstraints mpcConstraints;
    
    // Training state
    int trainingPhase;  // 0=warmup, 1=strategist, 2=pilot, 3=joint
    float strategistLoss;
    float pilotLoss;
    float totalReward;
};

// Update strategist (low frequency)
__device__ inline void updateStrategist(
    HierarchicalRLAgent* agent,
    float dt,
    curandState* randState
) {
    agent->strategistUpdateTimer += dt;
    
    if (agent->strategistUpdateTimer >= agent->strategistUpdateInterval) {
        agent->strategistUpdateTimer = 0.0f;
        
        // Encode current defense state
        float stateVector[STRATEGIST_STATE_DIM];
        encodeDefenseState(&agent->defenseState, stateVector);
        
        // Get strategist action
        float actionMean[STRATEGIST_ACTION_DIM];
        float actionStd[STRATEGIST_ACTION_DIM];
        float value;
        strategistForward(&agent->strategist, stateVector, actionMean, actionStd, &value);
        
        // Sample action
        float actionVector[STRATEGIST_ACTION_DIM];
        for (int i = 0; i < STRATEGIST_ACTION_DIM; i++) {
            float noise = curand_normal(randState);
            actionVector[i] = actionMean[i] + actionStd[i] * noise;
            actionVector[i] = fmaxf(-1.0f, fminf(1.0f, actionVector[i]));
        }
        
        // Decode to command
        agent->currentStrategy = decodeStrategistAction(actionVector, &agent->defenseState);
    }
}

// Update pilot for single interceptor (high frequency)
__device__ inline PilotAction updatePilot(
    HierarchicalRLAgent* agent,
    int interceptorIdx,
    EngagementState* engagement,
    float dt,
    curandState* randState
) {
    // Encode engagement state
    float stateVector[PILOT_STATE_DIM];
    encodeEngagementState(engagement, stateVector);
    
    // Get pilot policy action
    float actionMean[PILOT_ACTION_DIM];
    float actionLogStd[PILOT_ACTION_DIM];
    float sampledAction[PILOT_ACTION_DIM];
    pilotPolicyForward(&agent->pilot, stateVector, actionMean, actionLogStd, randState, sampledAction);
    
    // Create pilot action
    PilotAction action;
    action.pitchCommand = sampledAction[0];
    action.yawCommand = sampledAction[1];
    action.rollCommand = sampledAction[2];
    action.throttleCommand = (sampledAction[3] + 1.0f) * 0.5f;  // Map to [0,1]
    
    // Convert to acceleration command based on engagement urgency
    float urgency = agent->currentStrategy.engagementUrgency;
    float accelScale = urgency * agent->mpcConstraints.maxNormalAccel * GRAVITY;
    
    action.commandedAcceleration = make_float3(
        action.yawCommand * accelScale,
        action.pitchCommand * accelScale,
        action.throttleCommand * agent->mpcConstraints.maxAxialAccel * GRAVITY
    );
    
    // Refine with MPC
    MPCSolution mpcSol = solvePilotMPC(engagement, &action, &agent->mpcConstraints, dt);
    action.commandedAcceleration = mpcSol.commandedAccel;
    action.throttleCommand = mpcSol.throttle;
    
    // Compute commanded direction for thrust vectoring
    float accelMag = length(action.commandedAcceleration);
    if (accelMag > 0.01f) {
        action.commandedDirection = action.commandedAcceleration / accelMag;
    } else {
        action.commandedDirection = normalize(engagement->lineOfSight);
    }
    
    return action;
}

// ============================================================================
// REWARD FUNCTIONS
// ============================================================================

// Strategist reward: global defense effectiveness
__device__ inline float computeStrategistReward(
    DefenseState* prevState,
    DefenseState* newState,
    float dt
) {
    float reward = 0.0f;
    
    // Successful intercepts (major positive)
    int newIntercepts = newState->successfulIntercepts - prevState->successfulIntercepts;
    reward += newIntercepts * 100.0f;
    
    // Missed threats (major negative)
    int newMisses = newState->missedThreats - prevState->missedThreats;
    reward -= newMisses * 200.0f;
    
    // Resource efficiency (don't waste interceptors)
    float interceptorUsed = (float)(prevState->availableInterceptors - newState->availableInterceptors);
    reward -= interceptorUsed * 5.0f;  // Small penalty for using resources
    
    // Discrimination bonus (rewarded for waiting to confirm threats)
    reward += (newState->avgDiscriminationConf - prevState->avgDiscriminationConf) * 20.0f;
    
    // Time pressure (negative shaping as threats get closer)
    reward -= (30.0f - newState->timeUntilSaturation) * 0.1f * dt;
    
    return reward;
}

// Pilot reward: engagement effectiveness
__device__ inline float computePilotReward(
    EngagementState* state,
    bool interceptSuccessful,
    bool missionAborted
) {
    float reward = 0.0f;
    
    // Terminal rewards
    if (interceptSuccessful) {
        reward += 100.0f;
        // Bonus for early intercept (more time margin)
        reward += state->timeToGo * 5.0f;
    } else if (missionAborted) {
        reward -= 50.0f;
    }
    
    // Shaping rewards (dense)
    // ZEM reduction (key for PN-style guidance)
    float zemReward = -state->zeroEffortMiss * 0.01f;
    reward += zemReward;
    
    // Closing velocity (we want to be closing)
    if (state->rangeRate < 0.0f) {  // Closing
        reward += 0.5f;
    } else {
        reward -= 1.0f;  // Opening is bad
    }
    
    // LOS rate (should be minimized in terminal phase)
    float losRateMag = length(state->lineOfSightRate);
    if (state->inTerminalPhase) {
        reward -= losRateMag * 10.0f;  // Penalize LOS rotation in terminal
    }
    
    // Fuel efficiency
    reward -= (1.0f - state->fuel / 15.0f) * 0.1f;
    
    // Constraint violation penalty
    if (state->angleOfAttack > 0.9f * (M_PI / 4.0f)) {
        reward -= 5.0f;  // Near stall
    }
    
    return reward;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

__host__ inline void initHierarchicalRLAgent(HierarchicalRLAgent* agent) {
    memset(agent, 0, sizeof(HierarchicalRLAgent));
    
    // Xavier initialization
    auto xavierInit = [](float* weights, int fanIn, int fanOut, int size) {
        float scale = sqrtf(2.0f / (fanIn + fanOut));
        for (int i = 0; i < size; i++) {
            weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    };
    
    // Strategist initialization
    xavierInit(agent->strategist.sharedWeights1, STRATEGIST_STATE_DIM, 256, STRATEGIST_STATE_DIM * 256);
    xavierInit(agent->strategist.sharedWeights2, 256, 128, 256 * 128);
    xavierInit(agent->strategist.actorWeights, 128, STRATEGIST_ACTION_DIM, 128 * STRATEGIST_ACTION_DIM);
    xavierInit(agent->strategist.criticWeights, 128, 1, 128);
    
    for (int i = 0; i < STRATEGIST_ACTION_DIM; i++) {
        agent->strategist.actorLogStd[i] = -0.5f;  // Initial std ~0.6
    }
    
    // Pilot initialization
    xavierInit(agent->pilot.policyWeights1, PILOT_STATE_DIM, 128, PILOT_STATE_DIM * 128);
    xavierInit(agent->pilot.policyWeights2, 128, 64, 128 * 64);
    xavierInit(agent->pilot.policyMeanWeights, 64, PILOT_ACTION_DIM, 64 * PILOT_ACTION_DIM);
    xavierInit(agent->pilot.policyLogStdWeights, 64, PILOT_ACTION_DIM, 64 * PILOT_ACTION_DIM);
    
    xavierInit(agent->pilot.q1Weights1, PILOT_STATE_DIM + PILOT_ACTION_DIM, 128, 
               (PILOT_STATE_DIM + PILOT_ACTION_DIM) * 128);
    xavierInit(agent->pilot.q1Weights2, 128, 64, 128 * 64);
    xavierInit(agent->pilot.q1WeightsOut, 64, 1, 64);
    
    // Copy to Q2 and targets
    memcpy(agent->pilot.q2Weights1, agent->pilot.q1Weights1, sizeof(agent->pilot.q1Weights1));
    memcpy(agent->pilot.q2Weights2, agent->pilot.q1Weights2, sizeof(agent->pilot.q1Weights2));
    memcpy(agent->pilot.q2WeightsOut, agent->pilot.q1WeightsOut, sizeof(agent->pilot.q1WeightsOut));
    
    memcpy(agent->pilot.q1TargetWeights1, agent->pilot.q1Weights1, sizeof(agent->pilot.q1Weights1));
    memcpy(agent->pilot.q1TargetWeights2, agent->pilot.q1Weights2, sizeof(agent->pilot.q1Weights2));
    memcpy(agent->pilot.q1TargetWeightsOut, agent->pilot.q1WeightsOut, sizeof(agent->pilot.q1WeightsOut));
    
    memcpy(agent->pilot.q2TargetWeights1, agent->pilot.q2Weights1, sizeof(agent->pilot.q2Weights1));
    memcpy(agent->pilot.q2TargetWeights2, agent->pilot.q2Weights2, sizeof(agent->pilot.q2Weights2));
    memcpy(agent->pilot.q2TargetWeightsOut, agent->pilot.q2WeightsOut, sizeof(agent->pilot.q2WeightsOut));
    
    agent->pilot.logAlpha = 0.0f;
    agent->pilot.targetEntropy = -PILOT_ACTION_DIM;
    
    // MPC constraints
    agent->mpcConstraints.maxNormalAccel = 40.0f;  // 40g
    agent->mpcConstraints.maxAxialAccel = 10.0f;
    agent->mpcConstraints.maxAngleOfAttack = 30.0f * M_PI / 180.0f;
    agent->mpcConstraints.maxFinDeflection = 25.0f * M_PI / 180.0f;
    agent->mpcConstraints.maxFinRate = 100.0f * M_PI / 180.0f;  // deg/s
    agent->mpcConstraints.minAltitude = 100.0f;
    
    agent->strategistUpdateInterval = 0.5f;  // Update strategy every 0.5s
    agent->trainingPhase = 0;
}

#endif // HIERARCHICAL_RL_CUH
