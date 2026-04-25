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
// CHECK IF POSITION IS INSIDE PROTECTED TERRITORY
// ============================================================================
__device__ inline bool isInsideTerritory(float3 position, float3 defensePos) {
    float3 toPos = position - defensePos;
    float horizontalDist = sqrtf(toPos.x * toPos.x + toPos.z * toPos.z);
    return horizontalDist < TERRITORY_RADIUS;
}

// ============================================================================
// COUNT ACTIVE INTERCEPTORS TARGETING A MISSILE (OPTIMIZED)
// Uses cached interceptorsAssigned value instead of O(n) scan
// Only do full scan occasionally for validation
// ============================================================================
__device__ inline int countActiveInterceptors(Missile* missiles, int missileCount, int targetId) {
    // Fast path: use the cached value from the enemy missile
    // This avoids O(n) scan every frame for every enemy
    // The value is kept up-to-date when interceptors are created/destroyed
    return missiles[targetId].interceptorsAssigned;
}

// ============================================================================
// RADAR DETECTION AND INTERCEPTION KERNEL (PhysX-enhanced with retry logic)
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
    // Update radar stats (only once per missile)
    if (m.detectionTime == 0.0f) {
        m.detectionTime = globalTime - m.launchTime;
        if (m.targetInsideRadar) {
            atomicAdd(&metrics->insideRadarCount, 1);
        } else {
            atomicAdd(&metrics->outsideRadarCount, 1);
        }
    }
    
    // Count how many active interceptors are already targeting this missile
    int activeInterceptors = countActiveInterceptors(missiles, missileCount, idx);
    m.interceptorsAssigned = activeInterceptors;
    
    // Determine if we should launch an interceptor:
    // - First time: interceptAttempts == 0 and no active interceptor
    // - Retry: previous interceptor failed (activeInterceptors == 0) and we have budget
    int currentAttempts = m.interceptAttempts;
    bool shouldLaunch = false;
    
    if (currentAttempts == 0 && activeInterceptors == 0) {
        // First interceptor needed
        shouldLaunch = true;
    } else if (currentAttempts > 0 && currentAttempts < MAX_INTERCEPTORS_PER_TARGET && activeInterceptors == 0) {
        // Retry case: all previous interceptors failed/timed out
        shouldLaunch = true;
    }

    if (shouldLaunch && *interceptorCount < MAX_INTERCEPTORS) {
      // Use atomicCAS to safely claim the launch slot (only one thread wins)
      int expected = currentAttempts;
      int desired = currentAttempts + 1;
      int prevAttempts = atomicCAS(&m.interceptAttempts, expected, desired);
      
      // If CAS failed, another thread already launched - skip
      if (prevAttempts != expected) {
          return;
      }
      
      // Check if this is a retry (penalty applies)
      if (prevAttempts > 0) {
          atomicAdd(&metrics->totalRetryAttempts, 1);
          atomicAdd((int*)&metrics->retryPenaltyTotal, 
                    __float_as_int(PENALTY_PER_EXTRA_INTERCEPTOR));
      }
      
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
      if (interceptorIdx < MAX_MISSILES) {  // FIX: Check against MAX_MISSILES, not missileCount
        // Increment active interceptor count on the target enemy missile
        atomicAdd(&m.interceptorsAssigned, 1);
        
        Missile &interceptor = missiles[interceptorIdx];
        
        // Initialize PhysX state for interceptor
        interceptor.position = d_DEFENSE_STATION;
        interceptor.position.y = 50.0f;  // Launch from slightly above ground
        interceptor.prevPosition = interceptor.position;

        float3 toIntercept = normalize(interceptPoint - interceptor.position);
        
        // Ensure interceptor doesn't launch straight up
        if (toIntercept.y > 0.95f) {
            toIntercept.y = 0.8f;
            float horizScale = sqrtf(1.0f - 0.64f);
            float3 horizDir = interceptPoint - interceptor.position;
            horizDir.y = 0;
            if (length(horizDir) > 1.0f) {
                horizDir = normalize(horizDir);
            } else {
                horizDir = make_float3(1.0f, 0.0f, 0.0f);
            }
            toIntercept.x = horizDir.x * horizScale;
            toIntercept.z = horizDir.z * horizScale;
            toIntercept = normalize(toIntercept);
        }
        
        float initialSpeed = 250.0f;
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
        interceptor.launchPos = interceptor.position;
        
        // Physical properties
        interceptor.fuel = 20.0f;  // More fuel for longer pursuit
        interceptor.mass = INTERCEPTOR_MASS;
        interceptor.dragCoefficient = DRAG_COEFFICIENT;
        interceptor.liftCoefficient = LIFT_COEFFICIENT;
        interceptor.referenceArea = REFERENCE_AREA_INTERCEPTOR;
        
        interceptor.active = 1;
        interceptor.hit = 0;
        interceptor.type = INTERCEPTOR_MISSILE;
        interceptor.endState = MISSILE_ACTIVE;
        interceptor.lifetime = 0.0f;
        interceptor.targetMissileId = idx;
        interceptor.launchTime = globalTime;
        interceptor.detectionTime = globalTime - m.launchTime;
        interceptor.missileId = interceptorIdx + 10000;
        
        // Track this interceptor on the enemy missile (prevAttempts is 0-indexed)
        // Note: m.interceptAttempts was already atomically incremented above
        if (prevAttempts < MAX_INTERCEPTORS_PER_TARGET) {
            m.assignedInterceptorIds[prevAttempts] = interceptor.missileId;
        }
        
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
        interceptor.interceptorsAssigned = 0;
        interceptor.interceptAttempts = 0;
        interceptor.landedInTerritory = 0;
        
        interceptor.insideRadar = 1;
        interceptor.targetInsideRadar = m.targetInsideRadar;

        atomicAdd(&metrics->totalLaunched, 1);
        atomicAdd(&metrics->totalInterceptorsFired, 1);
        atomicAdd(&metrics->defenseMissilesLaunched, 1);
      }
    }
  }
}

// ============================================================================
// UPDATE MISSILE PHYSICS (PhysX-based with comprehensive tracking)
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
  float3 targetDir = (distToTarget > 1.0f) ? normalize(toTarget) : make_float3(0, -1, 0);
  
  // Track minimum distance to target
  if (distToTarget < m.minDistanceToTarget) {
      m.minDistanceToTarget = distToTarget;
  }

  float thrustMag = 0.0f;
  
  // ENEMY MISSILES: Ballistic trajectory - NO thrust, NO steering after launch
  // They follow pure ballistic physics: gravity + drag only
  if (m.type == ENEMY_MISSILE) {
      thrustMag = 0.0f;  // No thrust
      // targetDir should be velocity-aligned for correct drag calculation
      // but we DON'T want lift/steering - handle this in physics function
      float speed = length(m.velocity);
      if (speed > 1.0f) {
          targetDir = normalize(m.velocity);  // Body axis aligned with velocity
      } else {
          // If nearly stationary, point downward (falling)
          targetDir = make_float3(0.0f, -1.0f, 0.0f);
      }
  }
  // INTERCEPTORS: Active guidance with thrust
  else if (m.type == INTERCEPTOR_MISSILE) {
      thrustMag = (m.fuel > 0.0f) ? INTERCEPTOR_THRUST : 0.0f;
      
      // Proportional navigation toward target
      if (m.targetMissileId >= 0 && m.targetMissileId < count) {
          Missile &target = missiles[m.targetMissileId];
          if (target.active) {
              float3 pnAccel = physxProportionalNavigation(
                  m.position, m.velocity,
                  target.position, target.velocity,
                  4.0f  // Navigation gain
              );
              
              // Blend PN acceleration with current velocity direction
              float3 desiredDir = m.velocity + pnAccel * dt;
              if (length(desiredDir) > 1.0f) {
                  targetDir = normalize(desiredDir);
              }
              
              // Update predicted impact based on target motion
              float3 relPos = target.position - m.position;
              float3 relVel = target.velocity - m.velocity;
              float closingSpeed = -dot(relVel, normalize(relPos));
              if (closingSpeed > 10.0f) {
                  float timeToIntercept = length(relPos) / closingSpeed;
                  m.predictedImpact = target.position + target.velocity * timeToIntercept;
              }
          } else {
              // Target destroyed or inactive - self-destruct or continue to last known position
              targetDir = normalize(m.predictedImpact - m.position);
          }
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

  // Ground collision - CRITICAL FOR DEFENSE TRACKING
  if (m.position.y <= GROUND_LEVEL) {
    m.position.y = GROUND_LEVEL;
    m.velocity = make_float3(0, 0, 0);
    m.active = 0;
    m.hit = 1;

    if (m.type == ENEMY_MISSILE) {
      float3 toDefense = d_DEFENSE_STATION - m.position;
      float distToDefense = sqrtf(toDefense.x * toDefense.x + toDefense.z * toDefense.z);
      
      // Check if landed in protected territory
      bool inTerritory = isInsideTerritory(m.position, d_DEFENSE_STATION);
      m.landedInTerritory = inTerritory ? 1 : 0;
      
      if (inTerritory) {
          // CRITICAL: Enemy missile hit inside territory - defense failed!
          m.endState = MISSILE_GROUND_HIT_TERRITORY;
          atomicAdd(&metrics->enemyMissilesLandedTerritory, 1);
          atomicAdd(&metrics->interceptFail, 1);
          
          // Severe penalty for territory hit
          float penalty = PENALTY_COLLISION_MISS * 2.0f;  // Double penalty for territory hit
          atomicAdd((int*)&metrics->episodeReward, __float_as_int(penalty));
          atomicAdd((int*)&metrics->totalPenalties, __float_as_int(-penalty));
      } else {
          // Landed outside territory - less severe
          m.endState = MISSILE_GROUND_HIT_OUTSIDE;
          atomicAdd(&metrics->enemyMissilesLandedOutside, 1);
          
          // Only count as failure if target was inside radar but we missed
          if (m.targetInsideRadar && distToDefense < RADAR_RANGE) {
              atomicAdd(&metrics->interceptFail, 1);
              
              float rewardBreakdown[8];
              bool shouldCalculate;
              float penalty = computeMissReward(&m, d_DEFENSE_STATION, rewardBreakdown, &shouldCalculate);
              
              if (shouldCalculate) {
                  atomicAdd((int*)&metrics->episodeReward, __float_as_int(penalty));
                  atomicAdd((int*)&metrics->totalPenalties, __float_as_int(-penalty));
              }
          }
      }
      
      // Account for this enemy missile
      atomicAdd(&metrics->enemyMissilesAccountedFor, 1);
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
        
        // Update comprehensive tracking
        m.endState = MISSILE_INTERCEPTED;
        target.endState = MISSILE_INTERCEPTED;
        atomicAdd(&metrics->enemyMissilesIntercepted, 1);
        atomicAdd(&metrics->defenseMissilesHit, 1);
        atomicAdd(&metrics->enemyMissilesAccountedFor, 1);
        atomicAdd(&metrics->defenseMissilesAccountedFor, 1);
        
        // Decrement active interceptor count on target (interceptor completed its mission)
        atomicSub(&target.interceptorsAssigned, 1);
        
        // Apply retry penalty if multiple attempts were used
        if (target.interceptAttempts > 1) {
            float retryPenalty = PENALTY_PER_EXTRA_INTERCEPTOR * (target.interceptAttempts - 1);
            atomicAdd((int*)&metrics->episodeReward, __float_as_int(retryPenalty));
        }
      }
    }
  }

  // ========================================================================
  // BOUNDS AND TERMINATION CHECKS WITH COMPREHENSIVE TRACKING
  // ========================================================================
  
  bool outOfBounds = fabsf(m.position.x) > WORLD_SIZE || 
                     fabsf(m.position.z) > WORLD_SIZE ||
                     m.position.y > 50000.0f;
  bool timedOut = m.lifetime > INTERCEPTOR_TIMEOUT;
  
  // Check if missile is going up forever (physics bug detection)
  // A missile should not continuously gain altitude without thrust
  bool goingUpForever = false;
  if (m.type == ENEMY_MISSILE && m.fuel <= 0.0f && m.velocity.y > 50.0f && m.position.y > 5000.0f) {
      // Enemy missile with no fuel should not be climbing significantly
      // Apply corrective gravity if velocity is too upward
      goingUpForever = true;
  }
  
  // Apply corrective physics for missiles going up forever
  if (goingUpForever) {
      // Force stronger downward acceleration
      m.velocity.y -= GRAVITY * dt * 3.0f;  // Triple gravity correction
  }
  
  // ENEMY MISSILES: Only deactivate on ground hit or extreme out of bounds
  if (m.type == ENEMY_MISSILE) {
    if (outOfBounds && !m.hit) {
      m.active = 0;
      m.endState = MISSILE_OUT_OF_BOUNDS;
      atomicAdd(&metrics->enemyMissilesOutOfBounds, 1);
      atomicAdd(&metrics->enemyMissilesAccountedFor, 1);
    }
  } 
  // INTERCEPTORS: Can timeout, miss target, or go out of bounds
  else {
    if (timedOut && !m.hit) {
      m.active = 0;
      m.endState = MISSILE_INTERCEPTOR_TIMEOUT;
      atomicAdd(&metrics->defenseMissilesMissed, 1);
      atomicAdd(&metrics->defenseMissilesAccountedFor, 1);
      
      // Decrement active interceptor count on target
      if (m.targetMissileId >= 0 && m.targetMissileId < count) {
          atomicSub(&missiles[m.targetMissileId].interceptorsAssigned, 1);
      }
      
      // Mark this as a failed intercept attempt - target may need retry
      // The radar kernel will detect this and potentially launch another
    }
    else if (outOfBounds && !m.hit) {
      m.active = 0;
      m.endState = MISSILE_INTERCEPTOR_MISSED;
      atomicAdd(&metrics->defenseMissilesMissed, 1);
      atomicAdd(&metrics->defenseMissilesAccountedFor, 1);
      
      // Decrement active interceptor count on target
      if (m.targetMissileId >= 0 && m.targetMissileId < count) {
          atomicSub(&missiles[m.targetMissileId].interceptorsAssigned, 1);
      }
    }
    // Check if interceptor hit ground (missed target completely)
    else if (m.position.y <= GROUND_LEVEL && !m.hit) {
      m.active = 0;
      m.hit = 1;
      m.endState = MISSILE_INTERCEPTOR_MISSED;
      atomicAdd(&metrics->defenseMissilesMissed, 1);
      atomicAdd(&metrics->defenseMissilesAccountedFor, 1);
      
      // Decrement active interceptor count on target
      if (m.targetMissileId >= 0 && m.targetMissileId < count) {
          atomicSub(&missiles[m.targetMissileId].interceptorsAssigned, 1);
      }
    }
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
