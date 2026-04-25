#ifndef PHYSNODE_PINN_CUH
#define PHYSNODE_PINN_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"
#include "sixdof_dynamics.cuh"

// ============================================================================
// PHYSICS-EMBEDDED NEURAL ODE (PhysNODE) FOR TRAJECTORY PREDICTION
// Based on research document: "Physics-Informed Neural Networks (PINN)"
// 
// Architecture:
// 1. Encoder (Feature Extractor) - 1D CNN/LSTM on radar observations
// 2. Parameter Estimator - MLP to estimate unknown physical parameters
// 3. Differentiable Physics Solver - Integrates known dynamics with learned corrections
// ============================================================================

// ============================================================================
// CONSTANTS FOR PhysNODE
// ============================================================================

#define PHYSNODE_INPUT_DIM 18      // Extended state input
#define PHYSNODE_LATENT_DIM 32     // Latent representation dimension
#define PHYSNODE_OUTPUT_DIM 12     // Output: position, velocity, uncertainty
#define PHYSNODE_SEQUENCE_LEN 20   // Observation sequence length
#define PHYSNODE_CONV_CHANNELS 32  // 1D CNN channels

// Physics parameters estimated by network
struct EstimatedPhysicsParams {
    float liftToDrag;          // L/D ratio for gliders
    float ballisticCoef;       // Beta = m / (Cd * A)
    float bankAngle;           // Control bank angle for HGV
    float thrustLevel;         // Estimated thrust fraction
    float massEstimate;        // Estimated current mass
    float3 windEstimate;       // Estimated wind velocity
    float dragMultiplier;      // Correction to base drag model
    float liftMultiplier;      // Correction to base lift model
};

// Prediction output with uncertainty quantification
struct PhysNodePrediction {
    float3 position;           // Mean predicted position
    float3 velocity;           // Mean predicted velocity
    float3 positionUncertainty;  // Position variance (epistemic)
    float3 velocityUncertainty;  // Velocity variance
    float confidence;          // Overall prediction confidence
    float physicsResidual;     // How well prediction satisfies physics
    EstimatedPhysicsParams params;  // Estimated physical parameters
    float predictionHorizon;   // Time into future for this prediction
};

// ============================================================================
// 1D CONVOLUTIONAL ENCODER FOR SEQUENCE PROCESSING
// Processes time-series of radar observations
// ============================================================================

struct Conv1DLayer {
    float weights[PHYSNODE_INPUT_DIM * PHYSNODE_CONV_CHANNELS * 3];  // 3-tap filter
    float bias[PHYSNODE_CONV_CHANNELS];
};

struct EncoderNetwork {
    Conv1DLayer conv1;
    Conv1DLayer conv2;
    float fcWeights[PHYSNODE_CONV_CHANNELS * 4 * PHYSNODE_LATENT_DIM];  // After pooling
    float fcBias[PHYSNODE_LATENT_DIM];
};

// 1D Convolution operation
__device__ inline void conv1d(
    float* input,      // [seq_len, input_channels]
    int seqLen,
    int inputChannels,
    Conv1DLayer* layer,
    int outputChannels,
    float* output      // [seq_len-2, output_channels]
) {
    for (int t = 1; t < seqLen - 1; t++) {  // Valid padding
        for (int oc = 0; oc < outputChannels; oc++) {
            float sum = layer->bias[oc];
            for (int k = -1; k <= 1; k++) {  // 3-tap filter
                for (int ic = 0; ic < inputChannels; ic++) {
                    int inputIdx = (t + k) * inputChannels + ic;
                    int weightIdx = (k + 1) * inputChannels * outputChannels + ic * outputChannels + oc;
                    sum += input[inputIdx] * layer->weights[weightIdx];
                }
            }
            output[(t - 1) * outputChannels + oc] = fmaxf(0.0f, sum);  // ReLU
        }
    }
}

// Global average pooling
__device__ inline void globalAvgPool(
    float* input,
    int seqLen,
    int channels,
    float* output  // [channels]
) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            sum += input[t * channels + c];
        }
        output[c] = sum / (float)seqLen;
    }
}

// Encode observation sequence to latent representation
__device__ inline void encodeObservations(
    EncoderNetwork* encoder,
    float* observationSequence,  // [PHYSNODE_SEQUENCE_LEN, PHYSNODE_INPUT_DIM]
    float* latentVector          // Output: [PHYSNODE_LATENT_DIM]
) {
    // First convolution
    float conv1Out[(PHYSNODE_SEQUENCE_LEN - 2) * PHYSNODE_CONV_CHANNELS];
    conv1d(observationSequence, PHYSNODE_SEQUENCE_LEN, PHYSNODE_INPUT_DIM,
           &encoder->conv1, PHYSNODE_CONV_CHANNELS, conv1Out);
    
    // Second convolution
    float conv2Out[(PHYSNODE_SEQUENCE_LEN - 4) * PHYSNODE_CONV_CHANNELS];
    conv1d(conv1Out, PHYSNODE_SEQUENCE_LEN - 2, PHYSNODE_CONV_CHANNELS,
           &encoder->conv2, PHYSNODE_CONV_CHANNELS, conv2Out);
    
    // Global average pooling
    float pooled[PHYSNODE_CONV_CHANNELS];
    globalAvgPool(conv2Out, PHYSNODE_SEQUENCE_LEN - 4, PHYSNODE_CONV_CHANNELS, pooled);
    
    // Fully connected to latent
    for (int i = 0; i < PHYSNODE_LATENT_DIM; i++) {
        float sum = encoder->fcBias[i];
        for (int j = 0; j < PHYSNODE_CONV_CHANNELS; j++) {
            sum += pooled[j] * encoder->fcWeights[j * PHYSNODE_LATENT_DIM + i];
        }
        latentVector[i] = tanhf(sum);
    }
}

// ============================================================================
// PARAMETER ESTIMATION NETWORK
// Maps latent representation to physical parameters
// ============================================================================

struct ParameterEstimatorNetwork {
    float weights1[PHYSNODE_LATENT_DIM * 64];
    float bias1[64];
    float weights2[64 * 32];
    float bias2[32];
    float weightsOut[32 * 12];  // 12 physical parameters
    float biasOut[12];
};

__device__ inline EstimatedPhysicsParams estimatePhysicsParams(
    ParameterEstimatorNetwork* net,
    float* latentVector  // [PHYSNODE_LATENT_DIM]
) {
    // Hidden layer 1
    float hidden1[64];
    for (int i = 0; i < 64; i++) {
        float sum = net->bias1[i];
        for (int j = 0; j < PHYSNODE_LATENT_DIM; j++) {
            sum += latentVector[j] * net->weights1[j * 64 + i];
        }
        hidden1[i] = fmaxf(0.0f, sum);  // ReLU
    }
    
    // Hidden layer 2
    float hidden2[32];
    for (int i = 0; i < 32; i++) {
        float sum = net->bias2[i];
        for (int j = 0; j < 64; j++) {
            sum += hidden1[j] * net->weights2[j * 32 + i];
        }
        hidden2[i] = fmaxf(0.0f, sum);
    }
    
    // Output layer with appropriate activations
    float rawOutput[12];
    for (int i = 0; i < 12; i++) {
        float sum = net->biasOut[i];
        for (int j = 0; j < 32; j++) {
            sum += hidden2[j] * net->weightsOut[j * 12 + i];
        }
        rawOutput[i] = sum;
    }
    
    // Map outputs to physical parameters with appropriate constraints
    EstimatedPhysicsParams params;
    
    // L/D ratio: 0.5 to 5.0 (sigmoid scaled)
    params.liftToDrag = 0.5f + 4.5f / (1.0f + expf(-rawOutput[0]));
    
    // Ballistic coefficient: 100 to 10000 kg/m^2 (log scale)
    params.ballisticCoef = 100.0f * expf(3.0f * tanhf(rawOutput[1]));
    
    // Bank angle: -60 to +60 degrees
    params.bankAngle = 60.0f * M_PI / 180.0f * tanhf(rawOutput[2]);
    
    // Thrust level: 0 to 1
    params.thrustLevel = 1.0f / (1.0f + expf(-rawOutput[3]));
    
    // Mass estimate: 100 to 5000 kg
    params.massEstimate = 100.0f + 4900.0f / (1.0f + expf(-rawOutput[4]));
    
    // Wind estimate: -50 to +50 m/s per component
    params.windEstimate = make_float3(
        50.0f * tanhf(rawOutput[5]),
        50.0f * tanhf(rawOutput[6]),
        50.0f * tanhf(rawOutput[7])
    );
    
    // Drag/lift multipliers: 0.5 to 2.0
    params.dragMultiplier = 0.5f + 1.5f / (1.0f + expf(-rawOutput[8]));
    params.liftMultiplier = 0.5f + 1.5f / (1.0f + expf(-rawOutput[9]));
    
    return params;
}

// ============================================================================
// DIFFERENTIABLE PHYSICS SOLVER
// Integrates equations of motion with learned parameter corrections
// ============================================================================

// State derivative using estimated physics parameters
__device__ inline void physNodeDerivative(
    float3 position,
    float3 velocity,
    EstimatedPhysicsParams* params,
    float3* dPosition,
    float3* dVelocity
) {
    // Position derivative is just velocity
    *dPosition = velocity;
    
    // Velocity derivative from forces
    float altitude = position.y;
    float airDensity = getAirDensity(altitude);
    float speed = length(velocity);
    
    // Gravity
    float3 accel = make_float3(0.0f, -GRAVITY, 0.0f);
    
    if (speed > 1.0f) {
        float3 velDir = velocity / speed;
        
        // Drag: F_d = 0.5 * rho * V^2 * (m / beta)
        float dragForce = 0.5f * airDensity * speed * speed * 
                         params->massEstimate / params->ballisticCoef;
        dragForce *= params->dragMultiplier;  // Neural network correction
        float3 drag = velDir * (-dragForce / params->massEstimate);
        
        // Lift (for gliding vehicles)
        if (params->liftToDrag > 1.0f) {
            float liftMag = dragForce * params->liftToDrag * params->liftMultiplier;
            
            // Lift direction (perpendicular to velocity, in vertical plane)
            float3 liftDir = make_float3(-velDir.y * cosf(params->bankAngle), 
                                         velDir.x * cosf(params->bankAngle),
                                         sinf(params->bankAngle));
            // Normalize and ensure perpendicular to velocity
            liftDir = liftDir - velDir * dot(liftDir, velDir);
            float liftDirMag = length(liftDir);
            if (liftDirMag > 0.01f) {
                liftDir = liftDir / liftDirMag;
                float3 lift = liftDir * (liftMag / params->massEstimate);
                accel = accel + lift;
            }
        }
        
        accel = accel + drag;
    }
    
    // Wind effect
    // The velocity used for aero should be airspeed, not ground speed
    // This is a correction term
    float3 windCorrection = params->windEstimate * (-0.001f);  // Small wind effect
    accel = accel + windCorrection;
    
    // Thrust (if powered phase)
    if (params->thrustLevel > 0.01f) {
        float thrustAccel = params->thrustLevel * THRUST_FORCE / params->massEstimate;
        float3 thrustDir = (speed > 1.0f) ? velocity / speed : make_float3(0.0f, 1.0f, 0.0f);
        accel = accel + thrustDir * thrustAccel;
    }
    
    *dVelocity = accel;
}

// RK4 integration with estimated parameters
__device__ inline void integratePhysNode(
    float3* position,
    float3* velocity,
    EstimatedPhysicsParams* params,
    float dt
) {
    float3 k1_pos, k1_vel, k2_pos, k2_vel, k3_pos, k3_vel, k4_pos, k4_vel;
    float3 tempPos, tempVel;
    
    // k1
    physNodeDerivative(*position, *velocity, params, &k1_pos, &k1_vel);
    
    // k2
    tempPos = *position + k1_pos * (dt * 0.5f);
    tempVel = *velocity + k1_vel * (dt * 0.5f);
    physNodeDerivative(tempPos, tempVel, params, &k2_pos, &k2_vel);
    
    // k3
    tempPos = *position + k2_pos * (dt * 0.5f);
    tempVel = *velocity + k2_vel * (dt * 0.5f);
    physNodeDerivative(tempPos, tempVel, params, &k3_pos, &k3_vel);
    
    // k4
    tempPos = *position + k3_pos * dt;
    tempVel = *velocity + k3_vel * dt;
    physNodeDerivative(tempPos, tempVel, params, &k4_pos, &k4_vel);
    
    // Update
    *position = *position + (k1_pos + k2_pos * 2.0f + k3_pos * 2.0f + k4_pos) * (dt / 6.0f);
    *velocity = *velocity + (k1_vel + k2_vel * 2.0f + k3_vel * 2.0f + k4_vel) * (dt / 6.0f);
}

// ============================================================================
// UNCERTAINTY QUANTIFICATION NETWORK
// Outputs variance estimates alongside mean predictions
// ============================================================================

struct UncertaintyNetwork {
    float weights1[PHYSNODE_LATENT_DIM * 32];
    float bias1[32];
    float weightsOut[32 * 6];  // 6 variance outputs (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
    float biasOut[6];
};

__device__ inline void estimateUncertainty(
    UncertaintyNetwork* net,
    float* latentVector,
    float predictionTime,  // Uncertainty grows with time
    float3* positionVar,
    float3* velocityVar
) {
    // Hidden layer
    float hidden[32];
    for (int i = 0; i < 32; i++) {
        float sum = net->bias1[i];
        for (int j = 0; j < PHYSNODE_LATENT_DIM; j++) {
            sum += latentVector[j] * net->weights1[j * 32 + i];
        }
        hidden[i] = fmaxf(0.0f, sum);
    }
    
    // Output variances (softplus to ensure positive)
    float rawVar[6];
    for (int i = 0; i < 6; i++) {
        float sum = net->biasOut[i];
        for (int j = 0; j < 32; j++) {
            sum += hidden[j] * net->weightsOut[j * 6 + i];
        }
        rawVar[i] = logf(1.0f + expf(sum));  // Softplus
    }
    
    // Uncertainty grows with prediction horizon (roughly quadratic for position)
    float timeScale = 1.0f + predictionTime * predictionTime * 0.01f;
    float velTimeScale = 1.0f + predictionTime * 0.1f;
    
    *positionVar = make_float3(
        rawVar[0] * timeScale * 10000.0f,  // Scale to meters^2
        rawVar[1] * timeScale * 10000.0f,
        rawVar[2] * timeScale * 10000.0f
    );
    
    *velocityVar = make_float3(
        rawVar[3] * velTimeScale * 100.0f,  // Scale to (m/s)^2
        rawVar[4] * velTimeScale * 100.0f,
        rawVar[5] * velTimeScale * 100.0f
    );
}

// ============================================================================
// COMPLETE PhysNODE NETWORK STRUCTURE
// ============================================================================

struct PhysNodeNetwork {
    EncoderNetwork encoder;
    ParameterEstimatorNetwork paramEstimator;
    UncertaintyNetwork uncertainty;
    
    // Additional learned residual network (for non-modeled dynamics)
    float residualWeights1[PHYSNODE_LATENT_DIM * 32];
    float residualBias1[32];
    float residualWeights2[32 * 6];  // 3 position + 3 velocity corrections
    float residualBias2[6];
};

// ============================================================================
// MAIN PhysNODE PREDICTION FUNCTION
// ============================================================================

__device__ inline PhysNodePrediction physNodePredict(
    PhysNodeNetwork* net,
    float* observationSequence,  // [PHYSNODE_SEQUENCE_LEN, PHYSNODE_INPUT_DIM]
    float3 currentPosition,      // Latest known position
    float3 currentVelocity,      // Latest known velocity
    float predictionTime,        // How far into future to predict
    float dt                     // Integration timestep
) {
    PhysNodePrediction pred;
    memset(&pred, 0, sizeof(PhysNodePrediction));
    
    // Step 1: Encode observation sequence to latent representation
    float latent[PHYSNODE_LATENT_DIM];
    encodeObservations(&net->encoder, observationSequence, latent);
    
    // Step 2: Estimate physical parameters from latent
    pred.params = estimatePhysicsParams(&net->paramEstimator, latent);
    
    // Step 3: Integrate physics forward with estimated parameters
    float3 pos = currentPosition;
    float3 vel = currentVelocity;
    
    float elapsed = 0.0f;
    while (elapsed < predictionTime) {
        float stepDt = fminf(dt, predictionTime - elapsed);
        integratePhysNode(&pos, &vel, &pred.params, stepDt);
        elapsed += stepDt;
        
        // Ground collision check
        if (pos.y <= GROUND_LEVEL) {
            pos.y = GROUND_LEVEL;
            vel = make_float3(0.0f, 0.0f, 0.0f);
            break;
        }
    }
    
    // Step 4: Compute learned residual correction (for unmodeled dynamics)
    float hidden[32];
    for (int i = 0; i < 32; i++) {
        float sum = net->residualBias1[i];
        for (int j = 0; j < PHYSNODE_LATENT_DIM; j++) {
            sum += latent[j] * net->residualWeights1[j * 32 + i];
        }
        hidden[i] = tanhf(sum);
    }
    
    float residual[6];
    for (int i = 0; i < 6; i++) {
        float sum = net->residualBias2[i];
        for (int j = 0; j < 32; j++) {
            sum += hidden[j] * net->residualWeights2[j * 6 + i];
        }
        residual[i] = tanhf(sum);
    }
    
    // Apply residual (scaled by prediction time to prevent large corrections for short predictions)
    float residualScale = fminf(predictionTime / 5.0f, 1.0f);
    pos.x += residual[0] * 500.0f * residualScale;
    pos.y += residual[1] * 500.0f * residualScale;
    pos.z += residual[2] * 500.0f * residualScale;
    vel.x += residual[3] * 50.0f * residualScale;
    vel.y += residual[4] * 50.0f * residualScale;
    vel.z += residual[5] * 50.0f * residualScale;
    
    pred.position = pos;
    pred.velocity = vel;
    
    // Step 5: Estimate uncertainty
    estimateUncertainty(&net->uncertainty, latent, predictionTime,
                       &pred.positionUncertainty, &pred.velocityUncertainty);
    
    // Overall confidence (inverse of average variance)
    float avgVar = (pred.positionUncertainty.x + pred.positionUncertainty.y + 
                   pred.positionUncertainty.z) / 3.0f;
    pred.confidence = 1.0f / (1.0f + sqrtf(avgVar) / 1000.0f);
    
    pred.predictionHorizon = predictionTime;
    
    // Step 6: Compute physics residual (how well prediction satisfies dynamics)
    // This is used during training to enforce physics constraints
    float3 accelExpected;
    float3 dummy;
    physNodeDerivative(pred.position, pred.velocity, &pred.params, &dummy, &accelExpected);
    
    // Compare to finite-difference acceleration from trajectory
    float3 accelActual = (pred.velocity - currentVelocity) / predictionTime;
    float3 residualAccel = accelActual - accelExpected;
    pred.physicsResidual = length(residualAccel);
    
    return pred;
}

// ============================================================================
// PHYSICS-INFORMED LOSS FUNCTION
// L_total = L_data + λ_phy * L_physics + L_boundary
// ============================================================================

__device__ inline float computePhysNodeLoss(
    PhysNodePrediction pred,
    float3 actualPosition,      // Ground truth position
    float3 actualVelocity,      // Ground truth velocity
    float physicsWeight,        // λ_phy
    float boundaryWeight,       // Weight for boundary conditions
    float uncertaintyWeight     // Weight for uncertainty regularization
) {
    // Data loss: Negative log-likelihood with learned variance
    // L_data = 0.5 * (||x - x_pred||^2 / σ^2 + log(σ^2))
    float3 posError = actualPosition - pred.position;
    float3 velError = actualVelocity - pred.velocity;
    
    float dataLoss = 0.0f;
    dataLoss += 0.5f * (posError.x * posError.x / (pred.positionUncertainty.x + 1e-6f) + 
                        logf(pred.positionUncertainty.x + 1e-6f));
    dataLoss += 0.5f * (posError.y * posError.y / (pred.positionUncertainty.y + 1e-6f) + 
                        logf(pred.positionUncertainty.y + 1e-6f));
    dataLoss += 0.5f * (posError.z * posError.z / (pred.positionUncertainty.z + 1e-6f) + 
                        logf(pred.positionUncertainty.z + 1e-6f));
    
    dataLoss += 0.5f * (velError.x * velError.x / (pred.velocityUncertainty.x + 1e-6f));
    dataLoss += 0.5f * (velError.y * velError.y / (pred.velocityUncertainty.y + 1e-6f));
    dataLoss += 0.5f * (velError.z * velError.z / (pred.velocityUncertainty.z + 1e-6f));
    
    // Physics loss: Residual of equations of motion should be zero
    // R = dv/dt - (F_aero + F_gravity) / m
    float physicsLoss = physicsWeight * pred.physicsResidual * pred.physicsResidual;
    
    // Boundary loss: Position should be above ground
    float boundaryLoss = 0.0f;
    if (pred.position.y < GROUND_LEVEL) {
        boundaryLoss = boundaryWeight * (GROUND_LEVEL - pred.position.y) * (GROUND_LEVEL - pred.position.y);
    }
    
    // Uncertainty regularization: Prevent overconfident predictions
    float uncertaintyLoss = uncertaintyWeight * (
        -logf(pred.positionUncertainty.x + 1e-6f) - 
        logf(pred.positionUncertainty.y + 1e-6f) - 
        logf(pred.positionUncertainty.z + 1e-6f)
    );
    
    return dataLoss + physicsLoss + boundaryLoss + uncertaintyLoss;
}

// ============================================================================
// MONTE CARLO DROPOUT FOR BAYESIAN UNCERTAINTY
// Run multiple forward passes with dropout to estimate epistemic uncertainty
// ============================================================================

#define MC_DROPOUT_SAMPLES 10
#define DROPOUT_RATE 0.1f

__device__ inline PhysNodePrediction physNodePredictBayesian(
    PhysNodeNetwork* net,
    float* observationSequence,
    float3 currentPosition,
    float3 currentVelocity,
    float predictionTime,
    float dt,
    curandState* randState
) {
    // Accumulate predictions from multiple forward passes
    float3 posSum = make_float3(0.0f, 0.0f, 0.0f);
    float3 velSum = make_float3(0.0f, 0.0f, 0.0f);
    float3 posSumSq = make_float3(0.0f, 0.0f, 0.0f);
    float3 velSumSq = make_float3(0.0f, 0.0f, 0.0f);
    
    PhysNodePrediction avgPred;
    memset(&avgPred, 0, sizeof(PhysNodePrediction));
    
    for (int sample = 0; sample < MC_DROPOUT_SAMPLES; sample++) {
        // Apply dropout to latent representation
        float latent[PHYSNODE_LATENT_DIM];
        encodeObservations(&net->encoder, observationSequence, latent);
        
        // Dropout mask
        for (int i = 0; i < PHYSNODE_LATENT_DIM; i++) {
            if (curand_uniform(randState) < DROPOUT_RATE) {
                latent[i] = 0.0f;
            } else {
                latent[i] /= (1.0f - DROPOUT_RATE);  // Scale to maintain expectation
            }
        }
        
        // Forward pass with dropout
        EstimatedPhysicsParams params = estimatePhysicsParams(&net->paramEstimator, latent);
        
        float3 pos = currentPosition;
        float3 vel = currentVelocity;
        float elapsed = 0.0f;
        while (elapsed < predictionTime) {
            float stepDt = fminf(dt, predictionTime - elapsed);
            integratePhysNode(&pos, &vel, &params, stepDt);
            elapsed += stepDt;
            if (pos.y <= GROUND_LEVEL) break;
        }
        
        // Accumulate
        posSum = posSum + pos;
        velSum = velSum + vel;
        posSumSq.x += pos.x * pos.x;
        posSumSq.y += pos.y * pos.y;
        posSumSq.z += pos.z * pos.z;
        velSumSq.x += vel.x * vel.x;
        velSumSq.y += vel.y * vel.y;
        velSumSq.z += vel.z * vel.z;
        
        // Accumulate params
        avgPred.params.liftToDrag += params.liftToDrag;
        avgPred.params.ballisticCoef += params.ballisticCoef;
    }
    
    // Compute mean
    float invN = 1.0f / MC_DROPOUT_SAMPLES;
    avgPred.position = posSum * invN;
    avgPred.velocity = velSum * invN;
    avgPred.params.liftToDrag *= invN;
    avgPred.params.ballisticCoef *= invN;
    
    // Compute variance (epistemic uncertainty from model uncertainty)
    avgPred.positionUncertainty = make_float3(
        posSumSq.x * invN - avgPred.position.x * avgPred.position.x,
        posSumSq.y * invN - avgPred.position.y * avgPred.position.y,
        posSumSq.z * invN - avgPred.position.z * avgPred.position.z
    );
    avgPred.velocityUncertainty = make_float3(
        velSumSq.x * invN - avgPred.velocity.x * avgPred.velocity.x,
        velSumSq.y * invN - avgPred.velocity.y * avgPred.velocity.y,
        velSumSq.z * invN - avgPred.velocity.z * avgPred.velocity.z
    );
    
    // Confidence from variance
    float avgVar = (avgPred.positionUncertainty.x + avgPred.positionUncertainty.y + 
                   avgPred.positionUncertainty.z) / 3.0f;
    avgPred.confidence = 1.0f / (1.0f + sqrtf(avgVar) / 1000.0f);
    
    avgPred.predictionHorizon = predictionTime;
    
    return avgPred;
}

// ============================================================================
// INITIALIZATION FUNCTION
// ============================================================================

__host__ inline void initPhysNodeNetwork(PhysNodeNetwork* net) {
    // Xavier initialization for weights
    srand(time(NULL));
    
    auto xavierInit = [](float* weights, int fanIn, int fanOut, int size) {
        float scale = sqrtf(2.0f / (fanIn + fanOut));
        for (int i = 0; i < size; i++) {
            weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    };
    
    // Encoder
    xavierInit(net->encoder.conv1.weights, PHYSNODE_INPUT_DIM * 3, PHYSNODE_CONV_CHANNELS, 
               PHYSNODE_INPUT_DIM * PHYSNODE_CONV_CHANNELS * 3);
    xavierInit(net->encoder.conv2.weights, PHYSNODE_CONV_CHANNELS * 3, PHYSNODE_CONV_CHANNELS,
               PHYSNODE_CONV_CHANNELS * PHYSNODE_CONV_CHANNELS * 3);
    xavierInit(net->encoder.fcWeights, PHYSNODE_CONV_CHANNELS, PHYSNODE_LATENT_DIM,
               PHYSNODE_CONV_CHANNELS * 4 * PHYSNODE_LATENT_DIM);
    
    // Parameter estimator
    xavierInit(net->paramEstimator.weights1, PHYSNODE_LATENT_DIM, 64, PHYSNODE_LATENT_DIM * 64);
    xavierInit(net->paramEstimator.weights2, 64, 32, 64 * 32);
    xavierInit(net->paramEstimator.weightsOut, 32, 12, 32 * 12);
    
    // Uncertainty network
    xavierInit(net->uncertainty.weights1, PHYSNODE_LATENT_DIM, 32, PHYSNODE_LATENT_DIM * 32);
    xavierInit(net->uncertainty.weightsOut, 32, 6, 32 * 6);
    
    // Residual network
    xavierInit(net->residualWeights1, PHYSNODE_LATENT_DIM, 32, PHYSNODE_LATENT_DIM * 32);
    xavierInit(net->residualWeights2, 32, 6, 32 * 6);
    
    // Initialize biases to zero
    memset(net->encoder.conv1.bias, 0, PHYSNODE_CONV_CHANNELS * sizeof(float));
    memset(net->encoder.conv2.bias, 0, PHYSNODE_CONV_CHANNELS * sizeof(float));
    memset(net->encoder.fcBias, 0, PHYSNODE_LATENT_DIM * sizeof(float));
    memset(net->paramEstimator.bias1, 0, 64 * sizeof(float));
    memset(net->paramEstimator.bias2, 0, 32 * sizeof(float));
    memset(net->paramEstimator.biasOut, 0, 12 * sizeof(float));
    memset(net->uncertainty.bias1, 0, 32 * sizeof(float));
    memset(net->uncertainty.biasOut, 0, 6 * sizeof(float));
    memset(net->residualBias1, 0, 32 * sizeof(float));
    memset(net->residualBias2, 0, 6 * sizeof(float));
}

#endif // PHYSNODE_PINN_CUH
