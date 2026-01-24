#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "../core/globals.h"
#include "physx_engine.cuh"
#include "neural_net.cuh"

// ============================================================================
// CHECK IF TARGET IS INSIDE RADAR RANGE
// ============================================================================
__device__ inline bool isTargetInsideRadar(float3 targetPos, float3 defensePos) {
    float3 toTarget = targetPos - defensePos;
    float distance = length(toTarget);
    return distance < RADAR_RANGE;
}

// ============================================================================
// COMPUTE REWARD (ONLY FOR TARGETS INSIDE RADAR)
// ============================================================================
__device__ inline float computeInterceptionReward(
    Missile* interceptor,
    Missile* target,
    float3 defensePos,
    float* rewardBreakdown,  // Array of 8 floats for individual components
    bool* shouldCalculate
) {
    // Check if target was inside radar
    *shouldCalculate = isTargetInsideRadar(target->target, defensePos);
    
    if (!(*shouldCalculate)) {
        for (int i = 0; i < 8; i++) rewardBreakdown[i] = 0.0f;
        return 0.0f;
    }
    
    float totalReward = 0.0f;
    
    // 1. Base intercept reward
    rewardBreakdown[0] = REWARD_INTERCEPT_SUCCESS;
    totalReward += rewardBreakdown[0];
    
    // 2. Speed bonus (faster interception = higher reward)
    float responseTime = interceptor->lifetime;
    if (responseTime < 5.0f) {
        rewardBreakdown[1] = REWARD_FAST_INTERCEPT_BONUS * (1.0f - responseTime / 5.0f);
    } else {
        rewardBreakdown[1] = 0.0f;
    }
    totalReward += rewardBreakdown[1];
    
    // 3. Early detection bonus
    float detectionDelay = interceptor->launchTime - target->launchTime;
    if (detectionDelay < 2.0f) {
        rewardBreakdown[2] = REWARD_EARLY_DETECTION * (1.0f - detectionDelay / 2.0f);
    } else {
        rewardBreakdown[2] = 0.0f;
    }
    totalReward += rewardBreakdown[2];
    
    // 4. Prediction accuracy bonus
    float predError = interceptor->predictionError;
    if (predError < 500.0f) {
        rewardBreakdown[3] = REWARD_ACCURATE_PREDICTION * (1.0f - predError / 500.0f);
    } else {
        rewardBreakdown[3] = 0.0f;
    }
    totalReward += rewardBreakdown[3];
    
    // 5. Fuel efficiency (penalty for wasting fuel)
    float fuelEfficiency = (interceptor->fuel) / 15.0f;  // Initial fuel was 15
    rewardBreakdown[4] = fuelEfficiency > 0.2f ? 10.0f * fuelEfficiency : PENALTY_FUEL_WASTE;
    totalReward += rewardBreakdown[4];
    
    // 6. Distance from defense (closer intercepts are safer)
    float distFromDefense = length(interceptor->position - defensePos);
    if (distFromDefense > RADAR_RANGE * 0.8f) {
        rewardBreakdown[5] = 20.0f;  // Bonus for intercepting far from base
    } else {
        rewardBreakdown[5] = 10.0f * (distFromDefense / RADAR_RANGE);
    }
    totalReward += rewardBreakdown[5];
    
    // 7-8. Reserved for future use
    rewardBreakdown[6] = 0.0f;
    rewardBreakdown[7] = 0.0f;
    
    return totalReward;
}

__device__ inline float computeMissReward(
    Missile* target,
    float3 defensePos,
    float* rewardBreakdown,
    bool* shouldCalculate
) {
    *shouldCalculate = isTargetInsideRadar(target->target, defensePos);
    
    if (!(*shouldCalculate)) {
        for (int i = 0; i < 8; i++) rewardBreakdown[i] = 0.0f;
        return 0.0f;
    }
    
    float totalReward = 0.0f;
    
    // 1. Base miss penalty
    rewardBreakdown[0] = PENALTY_MISS;
    totalReward += rewardBreakdown[0];
    
    // 2. Additional penalty if hit defense station
    float distFromDefense = length(target->position - defensePos);
    if (distFromDefense < 500.0f) {
        rewardBreakdown[1] = PENALTY_COLLISION_MISS;
    } else {
        rewardBreakdown[1] = -10.0f * (1.0f - distFromDefense / RADAR_RANGE);
    }
    totalReward += rewardBreakdown[1];
    
    // 3-7. Reserved
    for (int i = 2; i < 8; i++) rewardBreakdown[i] = 0.0f;
    
    return totalReward;
}

// ============================================================================
// RADAR DETECTION AND INTERCEPTION KERNEL (PhysX-enhanced)
// ============================================================================
__global__ void radarDetectionKernel(Missile *missiles, int missileCount,
                                     RLAgent *agent, TrainingMetrics *metrics,
                                     int *interceptorCount, float dt,
                                     float globalTime) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= missileCount)
    return;

  Missile &m = missiles[idx];
  if (!m.active || m.type != ENEMY_MISSILE)
    return;

  float3 toMissile = m.position - d_DEFENSE_STATION;
  float distance = length(toMissile);
  
  // Track if target is inside radar
  m.insideRadar = (distance < RADAR_RANGE) ? 1 : 0;
  m.targetInsideRadar = isTargetInsideRadar(m.target, d_DEFENSE_STATION) ? 1 : 0;

  if (distance < RADAR_RANGE && distance > 500.0f) {
    // Update radar stats
    if (m.targetInsideRadar) {
        atomicAdd(&metrics->insideRadarCount, 1);
    } else {
        atomicAdd(&metrics->outsideRadarCount, 1);
    }
    
    bool alreadyTargeted = false;
    for (int i = 0; i < missileCount; i++) {
      if (missiles[i].active && missiles[i].type == INTERCEPTOR_MISSILE &&
          missiles[i].targetMissileId == idx) {
        alreadyTargeted = true;
        break;
      }
    }

    if (!alreadyTargeted && *interceptorCount < MAX_INTERCEPTORS) {
      // Enhanced PINN prediction
      float3 predictedPos;
      float confidence;
      pinnPredictTrajectory(&agent->predictionNet, m.position, m.velocity,
                            m.acceleration, m.fuel, m.altitude, 2.0f, 20,
                            &predictedPos, &confidence);
      
      m.predictedImpact = predictedPos;

      float interceptorSpeed = 800.0f;
      
      // PhysX-enhanced intercept calculation
      float3 interceptPoint = physxCalculateInterceptPoint(
          m.position, m.velocity, d_DEFENSE_STATION, interceptorSpeed,
          m.fuel, m.target);

      float launchAngle, launchPitch;
      getPolicyAction(&agent->policyNet, m.position, m.velocity,
                      d_DEFENSE_STATION, &launchAngle, &launchPitch);

      int interceptorIdx = atomicAdd(interceptorCount, 1);
      if (interceptorIdx < missileCount) {
        Missile &interceptor = missiles[interceptorIdx];
        
        // Initialize PhysX state for interceptor
        interceptor.position = d_DEFENSE_STATION;
        interceptor.prevPosition = d_DEFENSE_STATION;

        float3 toIntercept = normalize(interceptPoint - d_DEFENSE_STATION);
        float initialSpeed = 200.0f;
        interceptor.velocity = toIntercept * initialSpeed;
        interceptor.prevVelocity = interceptor.velocity;
        interceptor.initialVelocity = interceptor.velocity;

        interceptor.acceleration = make_float3(0, 0, 0);
        interceptor.angularVelocity = make_float3(0, 0, SPIN_RATE);
        interceptor.orientation = make_float3(0, 0, 0);
        interceptor.force = make_float3(0, 0, 0);
        interceptor.torque = make_float3(0, 0, 0);
        
        interceptor.target = interceptPoint;
        interceptor.predictedImpact = predictedPos;
        interceptor.launchPos = d_DEFENSE_STATION;
        
        // Physical properties
        interceptor.fuel = 15.0f;
        interceptor.mass = INTERCEPTOR_MASS;
        interceptor.dragCoefficient = DRAG_COEFFICIENT;
        interceptor.liftCoefficient = LIFT_COEFFICIENT;
        interceptor.referenceArea = REFERENCE_AREA_INTERCEPTOR;
        
        interceptor.active = 1;
        interceptor.hit = 0;
        interceptor.type = INTERCEPTOR_MISSILE;
        interceptor.lifetime = 0.0f;
        interceptor.targetMissileId = idx;
        interceptor.launchTime = globalTime;
        interceptor.detectionTime = globalTime - m.launchTime;
        interceptor.missileId = interceptorIdx + 10000;
        
        // Statistics initialization
        interceptor.maxAltitude = 0.0f;
        interceptor.maxVelocity = initialSpeed;
        interceptor.distanceTraveled = 0.0f;
        interceptor.fuelConsumed = 0.0f;
        interceptor.averageThrust = 0.0f;
        interceptor.maxAcceleration = 0.0f;
        interceptor.minDistanceToTarget = distance;
        interceptor.predictionError = 0.0f;
        interceptor.substepAccumulator = 0.0f;
        
        interceptor.insideRadar = 1;
        interceptor.targetInsideRadar = m.targetInsideRadar;

        atomicAdd(&metrics->totalLaunched, 1);
        atomicAdd(&metrics->totalInterceptorsFired, 1);
      }
    }
  }
}

// ============================================================================
// UPDATE MISSILE PHYSICS (PhysX-based)
// ============================================================================
__global__ void updateMissilesKernel(Missile *missiles, int count, float dt,
                                     TrainingMetrics *metrics) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  Missile &m = missiles[idx];
  if (!m.active)
    return;

  // Calculate target direction and thrust
  float3 toTarget = m.target - m.position;
  float distToTarget = length(toTarget);
  float3 targetDir = (distToTarget > 1.0f) ? normalize(toTarget) : make_float3(0, 1, 0);
  
  // Track minimum distance to target
  if (distToTarget < m.minDistanceToTarget) {
      m.minDistanceToTarget = distToTarget;
  }

  float thrustMag = (m.type == INTERCEPTOR_MISSILE) ? INTERCEPTOR_THRUST : 0.0f;
  if (m.type == ENEMY_MISSILE) {
      float speed = length(m.velocity);
      targetDir = (speed > 1.0f) ? normalize(m.velocity) : make_float3(0, 1, 0);
  }
  
  // For interceptors, use proportional navigation
  if (m.type == INTERCEPTOR_MISSILE && m.targetMissileId >= 0) {
      Missile &target = missiles[m.targetMissileId];
      if (target.active) {
          float3 pnAccel = physxProportionalNavigation(
              m.position, m.velocity,
              target.position, target.velocity,
              4.0f  // Navigation gain
          );
          targetDir = normalize(m.velocity + pnAccel * dt);
      }
  }

  // PhysX substep simulation
  physxSimulateSubsteps(&m, targetDir, thrustMag, dt);
  
  // Accumulate PhysX metrics
  float speed = length(m.velocity);
  atomicAdd((int*)&metrics->avgMachNumber, __float_as_int(m.machNumber));
  if (m.machNumber > metrics->maxMachNumber) {
      metrics->maxMachNumber = m.machNumber;
  }

  // Ground collision
  if (m.position.y <= GROUND_LEVEL) {
    m.position.y = GROUND_LEVEL;
    m.velocity = make_float3(0, 0, 0);
    m.active = 0;
    m.hit = 1;

    if (m.type == ENEMY_MISSILE) {
      float3 toDefense = d_DEFENSE_STATION - m.position;
      float distToDefense = length(toDefense);
      
      // Only count as failure if target was inside radar
      if (m.targetInsideRadar && distToDefense < RADAR_RANGE) {
        atomicAdd(&metrics->interceptFail, 1);
        
        // Calculate miss penalty
        float rewardBreakdown[8];
        bool shouldCalculate;
        float penalty = computeMissReward(&m, d_DEFENSE_STATION, rewardBreakdown, &shouldCalculate);
        
        if (shouldCalculate) {
            atomicAdd((int*)&metrics->episodeReward, __float_as_int(penalty));
            atomicAdd((int*)&metrics->totalPenalties, __float_as_int(-penalty));
        }
      }
    }
  }

  // Interceptor collision detection with PhysX
  if (m.type == INTERCEPTOR_MISSILE && m.targetMissileId >= 0) {
    Missile &target = missiles[m.targetMissileId];
    if (target.active && target.type == ENEMY_MISSILE) {
      float3 collisionPoint;
      bool collision = physxCheckCollision(
          m.position, COLLISION_RADIUS * 0.5f,
          target.position, COLLISION_RADIUS * 0.5f,
          m.velocity, target.velocity,
          dt, &collisionPoint
      );

      if (collision) {
        m.active = 0;
        m.hit = 1;
        target.active = 0;
        target.hit = 1;
        m.interceptTime = m.lifetime;
        
        // Calculate prediction error
        float3 predError = collisionPoint - m.predictedImpact;
        m.predictionError = length(predError);
        atomicAdd((int*)&metrics->avgPredictionError, __float_as_int(m.predictionError));

        // Calculate reward only if target was inside radar
        float rewardBreakdown[8];
        bool shouldCalculate;
        float reward = computeInterceptionReward(&m, &target, d_DEFENSE_STATION, 
                                                  rewardBreakdown, &shouldCalculate);
        
        if (shouldCalculate) {
            atomicAdd(&metrics->interceptSuccess, 1);
            atomicAdd((int*)&metrics->episodeReward, __float_as_int(reward));
            atomicAdd((int*)&metrics->totalInterceptReward, __float_as_int(rewardBreakdown[0]));
            atomicAdd((int*)&metrics->totalSpeedBonus, __float_as_int(rewardBreakdown[1]));
            atomicAdd((int*)&metrics->totalPredictionBonus, __float_as_int(rewardBreakdown[3]));
            atomicAdd((unsigned int*)&metrics->avgResponseTime, __float_as_uint(m.lifetime));
        }
      }
    }
  }

  // Bounds check
  if (fabsf(m.position.x) > WORLD_SIZE || fabsf(m.position.z) > WORLD_SIZE ||
      m.position.y > 50000.0f || m.lifetime > 120.0f) {
    m.active = 0;
  }
}

// ============================================================================
// TRAINING UPDATE KERNEL (PPO + SAC + MAML)
// ============================================================================
__global__ void updateRLAgentKernel(RLAgent *agent, TrainingMetrics *metrics,
                                    float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > 0)
    return;

  float successRate =
      (metrics->interceptSuccess + 1.0f) /
      (metrics->interceptSuccess + metrics->interceptFail + 2.0f);

  // Calculate total reward from components
  float reward = metrics->episodeReward;
  
  // Additional shaping based on metrics
  reward += successRate * 50.0f;
  reward -= metrics->avgResponseTime * 2.0f;
  
  // PINN accuracy bonus
  if (metrics->avgPredictionError < 200.0f) {
      reward += (200.0f - metrics->avgPredictionError) * 0.1f;
  }

  agent->totalReward += reward;
  agent->movingAvgReward = agent->movingAvgReward * 0.99f + reward * 0.01f;

  if (successRate > agent->bestSuccessRate) {
    agent->bestSuccessRate = successRate;
  }

  // Update recent statistics
  agent->recentRewards[agent->recentIndex] = reward;
  agent->recentSuccessRates[agent->recentIndex] = successRate;
  agent->recentIndex = (agent->recentIndex + 1) % 100;

  // PPO update schedule
  if (agent->episodeCount % PPO_EPOCHS == 0 && agent->ppoBuffer.count > 0) {
    // Compute GAE
    computeGAE(&agent->ppoBuffer, agent->gamma, agent->gaeLambda);
    
    // Reset buffer
    agent->ppoBuffer.count = 0;
  }

  // Learning rate decay
  if (agent->episodeCount % 100 == 0) {
    agent->learningRate *= 0.99f;
    agent->epsilon *= 0.995f;
    agent->entropyCoef *= 0.999f;
  }
  
  // Adaptive exploration based on performance
  if (successRate < 0.3f) {
      agent->epsilon = fminf(agent->epsilon * 1.01f, 0.5f);
  } else if (successRate > 0.7f) {
      agent->epsilon = fmaxf(agent->epsilon * 0.99f, 0.01f);
  }

  // MAML meta-update check
  if (agent->mamlState.currentTask >= MAML_META_BATCH_SIZE) {
      mamlMetaUpdate(&agent->mamlState, &agent->policyNet, MAML_OUTER_LR);
      metrics->mamlMetaLoss = 0.0f;
      for (int i = 0; i < MAML_META_BATCH_SIZE; i++) {
          metrics->mamlMetaLoss += agent->mamlState.taskRewards[i];
      }
      metrics->mamlMetaLoss /= MAML_META_BATCH_SIZE;
  }

  agent->episodeCount++;
  agent->updateStep++;

  // Exploration with curand
  if (agent->epsilon > 0.01f) {
    curandState state;
    curand_init(clock64(), idx, 0, &state);

    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
      if (curand_uniform(&state) < agent->epsilon * 0.1f) {
        agent->policyNet.weights1[i] += (curand_uniform(&state) - 0.5f) * 0.01f;
        agent->predictionNet.weights1[i] += (curand_uniform(&state) - 0.5f) * 0.005f;
      }
    }
    
    // Also update actor-critic weights occasionally
    if (curand_uniform(&state) < 0.1f) {
        for (int i = 0; i < NEURAL_HIDDEN * 6; i++) {
            if (curand_uniform(&state) < agent->epsilon) {
                agent->actorCritic.actorWeights[i] += (curand_uniform(&state) - 0.5f) * 0.005f;
            }
        }
    }
  }
}

#endif // KERNELS_CUH
