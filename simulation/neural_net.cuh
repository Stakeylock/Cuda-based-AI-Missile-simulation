#ifndef NEURAL_NET_CUH
#define NEURAL_NET_CUH

#include "cuda_utils.cuh"
#include "physx_engine.cuh"

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================
__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-fminf(fmaxf(x, -20.0f), 20.0f)));
}

__device__ inline float softplus(float x) {
    return logf(1.0f + expf(fminf(x, 20.0f)));
}

__device__ inline float leakyRelu(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

__device__ inline float swish(float x) {
    return x * sigmoid(x);
}

// ============================================================================
// PHYSICS-INFORMED NEURAL NETWORK (PINN) - ENHANCED
// With true physics constraints and uncertainty estimation
// ============================================================================

// Structure to hold PINN prediction with uncertainty
struct PINNPrediction {
    float3 position;       // Predicted position (mean)
    float3 velocity;       // Predicted velocity (mean)
    float3 positionVar;    // Position variance (uncertainty)
    float3 velocityVar;    // Velocity variance (uncertainty)
    float confidence;      // Overall confidence score
    float physicsResidual; // Physics constraint violation magnitude
    float dataLoss;        // Data-fitting loss component
    float physicsLoss;     // Physics constraint loss component
};

// Compute physics residual: F = ma should hold
// This is the key constraint embedded in PINN training
__device__ inline float computePhysicsResidual(
    float3 pos, float3 vel, float3 accel,
    float3 predictedAccel,
    float mass, float fuel, float dragCoef, float refArea
) {
    // Expected acceleration from physics
    float altitude = pos.y;
    float airDens = getAirDensity(altitude);
    float speed = length(vel);
    
    // 1. Gravity force
    float3 gravityAccel = make_float3(0.0f, -GRAVITY, 0.0f);
    
    // 2. Drag force: F_d = 0.5 * rho * v^2 * Cd * A
    float3 dragAccel = make_float3(0.0f, 0.0f, 0.0f);
    if (speed > 1.0f) {
        float mach = getMachNumber(speed, altitude);
        float Cd = getDragCoefficient(mach, dragCoef);
        float dragMag = 0.5f * airDens * speed * speed * Cd * refArea / mass;
        dragAccel = normalize(vel) * (-dragMag);
    }
    
    // 3. Thrust (simplified - along velocity direction)
    float3 thrustAccel = make_float3(0.0f, 0.0f, 0.0f);
    if (fuel > 0.0f) {
        thrustAccel = accel;  // Use provided acceleration as thrust component
    }
    
    // Total expected physics-based acceleration
    float3 physicsAccel = gravityAccel + dragAccel + thrustAccel;
    
    // Residual: difference between predicted and physics-expected
    float3 residual = predictedAccel - physicsAccel;
    
    return length(residual);
}

// Enhanced PINN trajectory prediction with uncertainty
__device__ inline PINNPrediction pinnPredictWithUncertainty(
    NeuralNetwork *net,
    float3 pos, float3 vel, float3 accel,
    float fuel, float altitude, float mass,
    float dragCoef, float refArea,
    float dt, int steps
) {
    PINNPrediction pred;
    memset(&pred, 0, sizeof(PINNPrediction));
    
    // Normalized input features (12-dimensional)
    float input[12] = {
        pos.x / 10000.0f, pos.y / 10000.0f, pos.z / 10000.0f,
        vel.x / 1000.0f, vel.y / 1000.0f, vel.z / 1000.0f,
        accel.x / 100.0f, accel.y / 100.0f, accel.z / 100.0f,
        fuel / 30.0f,
        altitude / 10000.0f,
        getMachNumber(length(vel), altitude) / 3.0f
    };

    // First hidden layer with physics-informed residual injection
    float hidden1[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->bias1[i];
        for (int j = 0; j < 6; j++) {
            sum += input[j] * net->weights1[j * NEURAL_HIDDEN + i];
        }
        // Inject physics residual (acceleration terms)
        sum += input[6 + (i % 3)] * 0.2f;  // Stronger physics coupling
        hidden1[i] = swish(sum);
    }

    // Second hidden layer
    float hidden2[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->bias2[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden1[j] * net->weights2[j * NEURAL_HIDDEN + i];
        }
        hidden2[i] = swish(sum);
    }

    // Output layer - 9 outputs:
    // [0-2]: Position correction mean
    // [3-5]: Velocity prediction mean  
    // [6-8]: Log variance (uncertainty) for position
    float output[9];
    for (int i = 0; i < 3; i++) {
        float sum = net->bias3[i % 3];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden2[j] * net->weights3[j * 3 + (i % 3)];
        }
        output[i] = tanh_activation(sum) * 1000.0f;  // Position correction (larger range)
    }
    
    // Velocity prediction
    for (int i = 3; i < 6; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden2[j] * net->weights2[j * NEURAL_HIDDEN + ((i-3) % NEURAL_HIDDEN)] * 0.1f;
        }
        output[i] = tanh_activation(sum) * 500.0f;
    }
    
    // Uncertainty estimation (log variance)
    for (int i = 6; i < 9; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden2[j] * 0.05f;
        }
        output[i] = softplus(sum);  // Ensure positive variance
    }

    // Physics-based trajectory prediction (kinematics)
    float totalDt = dt * steps;
    
    // Get expected physics acceleration
    float3 gravity = make_float3(0.0f, -GRAVITY, 0.0f);
    float airDens = getAirDensity(altitude);
    float speed = length(vel);
    
    float3 dragAccel = make_float3(0.0f, 0.0f, 0.0f);
    if (speed > 0.1f) {
        float mach = getMachNumber(speed, altitude);
        float Cd = getDragCoefficient(mach, dragCoef);
        float dragMag = 0.5f * airDens * speed * speed * Cd * refArea / mass;
        dragAccel = normalize(vel) * (-dragMag);
    }
    
    float3 physicsAccel = gravity + dragAccel;
    if (fuel > 0.0f) {
        physicsAccel = physicsAccel + accel;
    }
    
    // Physics-predicted position: s = s0 + v*t + 0.5*a*t^2
    float3 physicsPos = pos + vel * totalDt + physicsAccel * (0.5f * totalDt * totalDt);
    
    // Physics-predicted velocity: v = v0 + a*t
    float3 physicsVel = vel + physicsAccel * totalDt;
    
    // Neural network corrections (learned residual)
    float3 nnPosCorrection = make_float3(output[0], output[1], output[2]);
    float3 nnVelCorrection = make_float3(output[3], output[4], output[5]);
    
    // Uncertainty estimates
    pred.positionVar = make_float3(
        output[6] * 10000.0f,
        output[7] * 10000.0f,
        output[8] * 10000.0f
    );
    
    // Confidence based on inverse of uncertainty
    float avgVar = (output[6] + output[7] + output[8]) / 3.0f;
    pred.confidence = 1.0f / (1.0f + avgVar);
    
    // Final predictions: physics + weighted correction
    // Weight correction by confidence (less confident = trust physics more)
    pred.position = physicsPos + nnPosCorrection * pred.confidence;
    pred.velocity = physicsVel + nnVelCorrection * pred.confidence;
    
    // Compute physics residual for loss calculation
    float3 predictedAccel = (pred.velocity - vel) / totalDt;
    pred.physicsResidual = computePhysicsResidual(
        pos, vel, accel, predictedAccel,
        mass, fuel, dragCoef, refArea
    );
    
    // Physics loss (soft constraint)
    pred.physicsLoss = pred.physicsResidual * pred.physicsResidual;
    
    // Data loss would be computed during training with actual data
    pred.dataLoss = 0.0f;
    
    return pred;
}

// PINN training loss with physics constraints
__device__ inline float pinnComputeLoss(
    PINNPrediction pred,
    float3 actualPos,
    float3 actualVel,
    float physicsWeight,     // Lambda for physics constraint
    float uncertaintyWeight  // Weight for uncertainty regularization
) {
    // Data loss (MSE between prediction and actual)
    float3 posError = pred.position - actualPos;
    float3 velError = pred.velocity - actualVel;
    float dataLoss = (posError.x * posError.x + posError.y * posError.y + posError.z * posError.z) / 
                     (pred.positionVar.x + pred.positionVar.y + pred.positionVar.z + 1e-6f);
    dataLoss += (velError.x * velError.x + velError.y * velError.y + velError.z * velError.z) / 1e6f;
    
    // Physics constraint loss (residual should be zero)
    float physicsLoss = physicsWeight * pred.physicsLoss;
    
    // Uncertainty regularization (prevent overconfident predictions)
    float uncertaintyLoss = uncertaintyWeight * (
        logf(pred.positionVar.x + 1e-6f) + 
        logf(pred.positionVar.y + 1e-6f) + 
        logf(pred.positionVar.z + 1e-6f)
    );
    
    return dataLoss + physicsLoss + uncertaintyLoss;
}

// Legacy wrapper for compatibility
__device__ inline void pinnPredictTrajectory(
    NeuralNetwork *net,
    float3 pos, float3 vel, float3 accel,
    float fuel, float altitude,
    float dt, int steps,
    float3 *predictedPos,
    float *confidence
) {
    PINNPrediction pred = pinnPredictWithUncertainty(
        net, pos, vel, accel, fuel, altitude,
        MISSILE_MASS, DRAG_COEFFICIENT, REFERENCE_AREA_MISSILE,
        dt, steps
    );
    
    *predictedPos = pred.position;
    *confidence = pred.confidence;
}

// Legacy wrapper for compatibility
__device__ inline void predictTrajectory(NeuralNetwork *net, float3 pos,
                                         float3 vel, float dt,
                                         float3 *predictedPos) {
    float confidence;
    pinnPredictTrajectory(net, pos, vel, make_float3(0, 0, 0), 10.0f, pos.y, dt, 1, predictedPos, &confidence);
}

// ============================================================================
// PPO ACTOR-CRITIC FORWARD PASS
// ============================================================================
__device__ inline void ppoForward(
    ActorCriticNetwork *net,
    float *state,      // 12-dimensional state
    float *actionMean, // 6-dimensional action mean
    float *actionStd,  // 6-dimensional action std
    float *value       // Scalar value estimate
) {
    // Shared layers
    float shared1[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->sharedBias1[i];
        for (int j = 0; j < 12; j++) {
            sum += state[j] * net->sharedWeights1[j * NEURAL_HIDDEN + i];
        }
        shared1[i] = swish(sum);
    }
    
    float shared2[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->sharedBias2[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += shared1[j] * net->sharedWeights2[j * NEURAL_HIDDEN + i];
        }
        shared2[i] = swish(sum);
    }
    
    // Actor head - action mean
    for (int i = 0; i < 6; i++) {
        float sum = net->actorBias[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += shared2[j] * net->actorWeights[j * 6 + i];
        }
        actionMean[i] = tanh_activation(sum);
    }
    
    // Action standard deviation (learned)
    for (int i = 0; i < 6; i++) {
        actionStd[i] = expf(net->actorLogStd[i]);
        actionStd[i] = fmaxf(fminf(actionStd[i], 1.0f), 0.01f);  // Clamp
    }
    
    // Critic head - value
    float valueSum = net->criticBias[0];
    for (int j = 0; j < NEURAL_HIDDEN; j++) {
        valueSum += shared2[j] * net->criticWeights[j];
    }
    *value = valueSum;
}

// ============================================================================
// PPO ACTION SAMPLING
// ============================================================================
__device__ inline void ppoSampleAction(
    float *actionMean,
    float *actionStd,
    curandState *randState,
    float *action,
    float *logProb
) {
    *logProb = 0.0f;
    
    for (int i = 0; i < 6; i++) {
        // Sample from Gaussian
        float noise = curand_normal(randState);
        action[i] = actionMean[i] + actionStd[i] * noise;
        action[i] = fmaxf(fminf(action[i], 1.0f), -1.0f);  // Clamp
        
        // Log probability of this action
        float diff = action[i] - actionMean[i];
        *logProb -= 0.5f * (diff * diff) / (actionStd[i] * actionStd[i]);
        *logProb -= logf(actionStd[i]) + 0.5f * logf(2.0f * M_PI);
    }
}

// ============================================================================
// PPO LOSS COMPUTATION
// ============================================================================
__device__ inline float ppoPolicyLoss(
    float oldLogProb,
    float newLogProb,
    float advantage
) {
    float ratio = expf(newLogProb - oldLogProb);
    float clipRatio = fmaxf(fminf(ratio, 1.0f + PPO_CLIP_EPSILON), 1.0f - PPO_CLIP_EPSILON);
    return -fminf(ratio * advantage, clipRatio * advantage);
}

__device__ inline float ppoValueLoss(float value, float returnValue) {
    float diff = value - returnValue;
    return 0.5f * diff * diff;
}

__device__ inline float ppoEntropyBonus(float *actionStd) {
    float entropy = 0.0f;
    for (int i = 0; i < 6; i++) {
        // Entropy of Gaussian
        entropy += logf(actionStd[i]) + 0.5f * logf(2.0f * M_PI * 2.718281828f);
    }
    return entropy;
}

// ============================================================================
// SAC Q-NETWORK FORWARD
// ============================================================================
__device__ inline float sacQForward(
    NeuralNetwork *qNet,
    float *state,   // 12-dim
    float *action   // 6-dim
) {
    // Concatenate state and action for Q input
    float input[6];
    for (int i = 0; i < 6; i++) {
        input[i] = (state[i] + action[i % 6]) * 0.5f;
    }
    
    float hidden1[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = qNet->bias1[i];
        for (int j = 0; j < 6; j++) {
            sum += input[j] * qNet->weights1[j * NEURAL_HIDDEN + i];
        }
        hidden1[i] = relu(sum);
    }
    
    float hidden2[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = qNet->bias2[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden1[j] * qNet->weights2[j * NEURAL_HIDDEN + i];
        }
        hidden2[i] = relu(sum);
    }
    
    float qValue = qNet->bias3[0];
    for (int j = 0; j < NEURAL_HIDDEN; j++) {
        qValue += hidden2[j] * qNet->weights3[j * 3];
    }
    
    return qValue;
}

// ============================================================================
// MAML INNER LOOP UPDATE
// ============================================================================
__device__ inline void mamlInnerUpdate(
    MAMLState *maml,
    NeuralNetwork *baseNet,
    float *loss,
    float innerLR
) {
    // Copy base weights to fast weights for task-specific adaptation
    if (maml->innerStep == 0) {
        for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
            maml->fastWeights1[i] = baseNet->weights1[i];
        }
        for (int i = 0; i < NEURAL_HIDDEN; i++) {
            maml->fastBias1[i] = baseNet->bias1[i];
        }
        for (int i = 0; i < NEURAL_HIDDEN * NEURAL_HIDDEN; i++) {
            maml->fastWeights2[i] = baseNet->weights2[i];
        }
        for (int i = 0; i < NEURAL_HIDDEN; i++) {
            maml->fastBias2[i] = baseNet->bias2[i];
        }
        for (int i = 0; i < NEURAL_HIDDEN * 3; i++) {
            maml->fastWeights3[i] = baseNet->weights3[i];
        }
        for (int i = 0; i < 3; i++) {
            maml->fastBias3[i] = baseNet->bias3[i];
        }
    }
    
    // Gradient descent on fast weights (simplified - in practice use backprop)
    float gradScale = innerLR * (*loss);
    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
        maml->fastWeights1[i] -= gradScale * 0.001f * (maml->fastWeights1[i] > 0 ? 1.0f : -1.0f);
    }
    
    maml->innerStep++;
}

// ============================================================================
// MAML META GRADIENT ACCUMULATION
// ============================================================================
__device__ inline void mamlAccumulateMetaGrad(
    MAMLState *maml,
    NeuralNetwork *baseNet,
    float taskLoss
) {
    // Accumulate gradients for meta-update
    maml->taskRewards[maml->currentTask] = -taskLoss;  // Negative loss as reward
    
    // Simple gradient estimation (difference between adapted and base)
    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
        maml->metaGrad1[i] += (maml->fastWeights1[i] - baseNet->weights1[i]) * taskLoss;
    }
    for (int i = 0; i < NEURAL_HIDDEN * NEURAL_HIDDEN; i++) {
        maml->metaGrad2[i] += (maml->fastWeights2[i] - baseNet->weights2[i]) * taskLoss;
    }
    for (int i = 0; i < NEURAL_HIDDEN * 3; i++) {
        maml->metaGrad3[i] += (maml->fastWeights3[i] - baseNet->weights3[i]) * taskLoss;
    }
    
    maml->currentTask++;
}

// ============================================================================
// MAML META UPDATE
// ============================================================================
__device__ inline void mamlMetaUpdate(
    MAMLState *maml,
    NeuralNetwork *baseNet,
    float metaLR
) {
    if (maml->currentTask < MAML_META_BATCH_SIZE) return;
    
    // Apply meta-gradients to base network
    float scale = metaLR / (float)MAML_META_BATCH_SIZE;
    
    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
        baseNet->weights1[i] -= scale * maml->metaGrad1[i];
        maml->metaGrad1[i] = 0.0f;  // Reset
    }
    for (int i = 0; i < NEURAL_HIDDEN * NEURAL_HIDDEN; i++) {
        baseNet->weights2[i] -= scale * maml->metaGrad2[i];
        maml->metaGrad2[i] = 0.0f;
    }
    for (int i = 0; i < NEURAL_HIDDEN * 3; i++) {
        baseNet->weights3[i] -= scale * maml->metaGrad3[i];
        maml->metaGrad3[i] = 0.0f;
    }
    
    maml->currentTask = 0;
    maml->innerStep = 0;
}

// ============================================================================
// CALCULATE OPTIMAL INTERCEPT POINT (PhysX-enhanced with PINN)
// ============================================================================
__device__ inline float3 calculateInterceptPoint(float3 missilePos,
                                                 float3 missileVel,
                                                 float3 defensePos,
                                                 float interceptorSpeed) {
    // Use PhysX-enhanced calculation
    return physxCalculateInterceptPoint(
        missilePos, missileVel, defensePos, interceptorSpeed,
        25.0f, missilePos + missileVel * 10.0f
    );
}

// ============================================================================
// RL POLICY NETWORK (Enhanced)
// ============================================================================
__device__ inline void getPolicyAction(NeuralNetwork *net, float3 missilePos,
                                       float3 missileVel, float3 defensePos,
                                       float *launchAngle, float *launchPitch) {
    float input[6] = {(missilePos.x - defensePos.x) / 10000.0f,
                    (missilePos.y - defensePos.y) / 10000.0f,
                    (missilePos.z - defensePos.z) / 10000.0f,
                    missileVel.x / 1000.0f,
                    missileVel.y / 1000.0f,
                    missileVel.z / 1000.0f};

    float hidden1[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->bias1[i];
        for (int j = 0; j < 6; j++) {
            sum += input[j] * net->weights1[j * NEURAL_HIDDEN + i];
        }
        hidden1[i] = swish(sum);
    }

    float hidden2[NEURAL_HIDDEN];
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
        float sum = net->bias2[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden1[j] * net->weights2[j * NEURAL_HIDDEN + i];
        }
        hidden2[i] = swish(sum);
    }

    float output[3];
    for (int i = 0; i < 3; i++) {
        float sum = net->bias3[i];
        for (int j = 0; j < NEURAL_HIDDEN; j++) {
            sum += hidden2[j] * net->weights3[j * 3 + i];
        }
        output[i] = tanh_activation(sum);
    }

    *launchAngle = output[0] * M_PI;
    *launchPitch = output[1] * M_PI / 2.0f;
}

// ============================================================================
// GAE (Generalized Advantage Estimation) for PPO
// ============================================================================
__device__ inline void computeGAE(
    PPOBuffer *buffer,
    float gamma,
    float lambda
) {
    float lastGAE = 0.0f;
    float lastValue = buffer->values[buffer->count - 1];
    
    for (int t = buffer->count - 1; t >= 0; t--) {
        float nextValue = (t == buffer->count - 1) ? 0.0f : buffer->values[t + 1];
        float delta = buffer->rewards[t] + gamma * nextValue - buffer->values[t];
        lastGAE = delta + gamma * lambda * lastGAE;
        buffer->advantages[t] = lastGAE;
        buffer->returns[t] = buffer->advantages[t] + buffer->values[t];
    }
    
    // Normalize advantages
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < buffer->count; i++) {
        mean += buffer->advantages[i];
    }
    mean /= buffer->count;
    
    for (int i = 0; i < buffer->count; i++) {
        float diff = buffer->advantages[i] - mean;
        var += diff * diff;
    }
    var = sqrtf(var / buffer->count + 1e-8f);
    
    for (int i = 0; i < buffer->count; i++) {
        buffer->advantages[i] = (buffer->advantages[i] - mean) / var;
    }
}

#endif // NEURAL_NET_CUH
