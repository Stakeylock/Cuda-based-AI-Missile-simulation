#ifndef CURRICULUM_LEARNING_CUH
#define CURRICULUM_LEARNING_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"
#include "hierarchical_rl.cuh"
#include "sensor_simulation.cuh"

// ============================================================================
// CURRICULUM LEARNING AND ADVERSARIAL SELF-PLAY
// Based on research document: "Curriculum Learning and Reward Shaping"
//
// Phases:
// 1. Non-maneuvering targets, noise-free sensors - Learn basic PN
// 2. Maneuvering targets (weave, step turns) - Learn prediction
// 3. Multi-agent with decoys - Learn discrimination and prioritization
// 4. Adversarial self-play - Co-evolution for robust strategies
// ============================================================================

// ============================================================================
// CURRICULUM PHASE DEFINITIONS
// ============================================================================

enum CurriculumPhase {
    PHASE_BASIC_PN = 0,           // Basic Proportional Navigation
    PHASE_MANEUVERING = 1,        // Maneuvering targets
    PHASE_MULTI_AGENT = 2,        // Multiple threats + decoys
    PHASE_ADVERSARIAL = 3,        // Self-play
    PHASE_FULL_COMPLEXITY = 4     // All challenges combined
};

// Difficulty parameters for each curriculum phase
struct CurriculumDifficulty {
    // Target behavior
    float maxTargetManeuver;       // Max g-load for target evasion
    float maneuverFrequency;       // How often targets change maneuver
    float maneuverRandomness;      // 0=predictable, 1=chaotic
    
    // Sensor quality
    float sensorNoiseLevel;        // Noise stddev multiplier
    float sensorUpdateRate;        // How often we get measurements
    float sensorDropoutProb;       // Probability of missed detections
    
    // Scenario complexity
    int maxThreats;                // Max simultaneous threats
    int maxDecoys;                 // Max decoys per wave
    float decoyQuality;            // How realistic decoys are (0=obvious, 1=perfect)
    
    // Time pressure
    float engagementRange;         // Initial range (closer = less time)
    float targetSpeedMultiplier;   // Speed of incoming threats
    
    // Resources
    int availableInterceptors;     // Interceptor budget
    float interceptorReliability;  // P(interceptor works)
};

// Curriculum state tracking
struct CurriculumState {
    CurriculumPhase currentPhase;
    int episodesInPhase;
    int totalEpisodes;
    
    // Performance tracking for advancement
    float recentSuccessRates[100];
    int recentIndex;
    float movingAvgSuccessRate;
    float bestSuccessRate;
    
    // Phase-specific metrics
    float avgInterceptTime;
    float avgPredictionError;
    float avgDiscriminationAccuracy;
    
    // Advancement thresholds
    float advancementThreshold;    // Success rate to advance
    int minEpisodesBeforeAdvance;  // Minimum episodes before considering
    
    // Current difficulty
    CurriculumDifficulty difficulty;
};

// ============================================================================
// CURRICULUM DIFFICULTY PRESETS
// ============================================================================

__host__ __device__ inline CurriculumDifficulty getPhaseBasicPN() {
    CurriculumDifficulty d;
    d.maxTargetManeuver = 0.0f;       // Non-maneuvering
    d.maneuverFrequency = 0.0f;
    d.maneuverRandomness = 0.0f;
    d.sensorNoiseLevel = 0.0f;        // Perfect sensors
    d.sensorUpdateRate = 100.0f;      // 100 Hz
    d.sensorDropoutProb = 0.0f;       // No dropouts
    d.maxThreats = 1;
    d.maxDecoys = 0;
    d.decoyQuality = 0.0f;
    d.engagementRange = 10000.0f;     // Long range (easy)
    d.targetSpeedMultiplier = 0.8f;   // Slower targets
    d.availableInterceptors = 5;
    d.interceptorReliability = 1.0f;
    return d;
}

__host__ __device__ inline CurriculumDifficulty getPhaseManeuvering() {
    CurriculumDifficulty d;
    d.maxTargetManeuver = 5.0f;       // 5g maneuvers
    d.maneuverFrequency = 0.2f;       // Change every 5 seconds
    d.maneuverRandomness = 0.3f;      // Somewhat predictable
    d.sensorNoiseLevel = 0.1f;        // Slight noise
    d.sensorUpdateRate = 50.0f;       // 50 Hz
    d.sensorDropoutProb = 0.01f;      // 1% dropout
    d.maxThreats = 3;
    d.maxDecoys = 0;
    d.decoyQuality = 0.0f;
    d.engagementRange = 8000.0f;
    d.targetSpeedMultiplier = 1.0f;
    d.availableInterceptors = 6;
    d.interceptorReliability = 0.95f;
    return d;
}

__host__ __device__ inline CurriculumDifficulty getPhaseMultiAgent() {
    CurriculumDifficulty d;
    d.maxTargetManeuver = 8.0f;       // 8g maneuvers
    d.maneuverFrequency = 0.3f;
    d.maneuverRandomness = 0.5f;
    d.sensorNoiseLevel = 0.2f;
    d.sensorUpdateRate = 30.0f;
    d.sensorDropoutProb = 0.05f;
    d.maxThreats = 10;
    d.maxDecoys = 20;
    d.decoyQuality = 0.5f;            // Moderate decoy quality
    d.engagementRange = 6000.0f;
    d.targetSpeedMultiplier = 1.2f;
    d.availableInterceptors = 8;
    d.interceptorReliability = 0.9f;
    return d;
}

__host__ __device__ inline CurriculumDifficulty getPhaseAdversarial() {
    CurriculumDifficulty d;
    d.maxTargetManeuver = 10.0f;      // 10g maneuvers
    d.maneuverFrequency = 0.5f;       // Frequent changes
    d.maneuverRandomness = 0.7f;      // Highly unpredictable
    d.sensorNoiseLevel = 0.3f;
    d.sensorUpdateRate = 20.0f;
    d.sensorDropoutProb = 0.1f;
    d.maxThreats = 20;
    d.maxDecoys = 40;
    d.decoyQuality = 0.8f;            // High quality decoys
    d.engagementRange = 5000.0f;
    d.targetSpeedMultiplier = 1.5f;
    d.availableInterceptors = 10;
    d.interceptorReliability = 0.85f;
    return d;
}

__host__ __device__ inline CurriculumDifficulty getPhaseFullComplexity() {
    CurriculumDifficulty d;
    d.maxTargetManeuver = 15.0f;      // Max maneuvers
    d.maneuverFrequency = 1.0f;       // Continuous maneuvering
    d.maneuverRandomness = 1.0f;
    d.sensorNoiseLevel = 0.5f;
    d.sensorUpdateRate = 10.0f;       // Sparse updates
    d.sensorDropoutProb = 0.15f;
    d.maxThreats = 50;
    d.maxDecoys = 150;
    d.decoyQuality = 1.0f;            // Perfect decoys (rely on physics)
    d.engagementRange = 4000.0f;
    d.targetSpeedMultiplier = 2.0f;   // HGV speeds
    d.availableInterceptors = 15;
    d.interceptorReliability = 0.8f;
    return d;
}

// ============================================================================
// TARGET MANEUVER PATTERNS
// ============================================================================

enum ManeuverType {
    MANEUVER_STRAIGHT = 0,
    MANEUVER_WEAVE_SINE = 1,
    MANEUVER_WEAVE_SQUARE = 2,
    MANEUVER_BARREL_ROLL = 3,
    MANEUVER_CORKSCREW = 4,
    MANEUVER_RANDOM_WALK = 5,
    MANEUVER_STEP_TURN = 6,
    MANEUVER_JINK = 7,
    MANEUVER_ADAPTIVE = 8  // Learns to evade (for self-play)
};

struct ManeuverState {
    ManeuverType type;
    float time;
    float frequency;      // Hz for periodic maneuvers
    float amplitude;      // G-load
    float phase;          // Phase offset
    float3 direction;     // Current maneuver direction
    int stepCount;        // For step maneuvers
    
    // For adaptive/learning maneuvers
    float3 observedInterceptorPos;
    float3 observedInterceptorVel;
    bool hasInterceptorInfo;
};

// Generate maneuver acceleration based on pattern
__device__ inline float3 generateManeuverAccel(
    ManeuverState* state,
    float3 position,
    float3 velocity,
    float dt,
    curandState* randState
) {
    state->time += dt;
    float3 accel = make_float3(0.0f, 0.0f, 0.0f);
    float g = GRAVITY * state->amplitude;
    
    switch (state->type) {
        case MANEUVER_STRAIGHT:
            // No maneuver acceleration
            break;
            
        case MANEUVER_WEAVE_SINE: {
            // Sinusoidal weave in horizontal plane
            float phase = 2.0f * M_PI * state->frequency * state->time + state->phase;
            float3 velDir = normalize(velocity);
            float3 up = make_float3(0.0f, 1.0f, 0.0f);
            float3 lateral = normalize(cross(velDir, up));
            accel = lateral * (g * sinf(phase));
            break;
        }
        
        case MANEUVER_WEAVE_SQUARE: {
            // Square wave weave
            float period = 1.0f / state->frequency;
            int halfPeriods = (int)(state->time / (period * 0.5f));
            float sign = (halfPeriods % 2 == 0) ? 1.0f : -1.0f;
            float3 velDir = normalize(velocity);
            float3 up = make_float3(0.0f, 1.0f, 0.0f);
            float3 lateral = normalize(cross(velDir, up));
            accel = lateral * (g * sign);
            break;
        }
        
        case MANEUVER_BARREL_ROLL: {
            // Roll while maintaining course
            float phase = 2.0f * M_PI * state->frequency * state->time + state->phase;
            float3 velDir = normalize(velocity);
            float3 up = make_float3(0.0f, 1.0f, 0.0f);
            float3 lateral = normalize(cross(velDir, up));
            accel = lateral * (g * sinf(phase)) + up * (g * cosf(phase) * 0.5f);
            break;
        }
        
        case MANEUVER_CORKSCREW: {
            // Helical evasion
            float phase = 2.0f * M_PI * state->frequency * state->time + state->phase;
            float3 velDir = normalize(velocity);
            float3 up = make_float3(0.0f, 1.0f, 0.0f);
            float3 lateral = normalize(cross(velDir, up));
            float3 vertical = normalize(cross(lateral, velDir));
            accel = lateral * (g * cosf(phase)) + vertical * (g * sinf(phase));
            break;
        }
        
        case MANEUVER_RANDOM_WALK: {
            // Random acceleration changes
            if (((float)rand() / RAND_MAX) < state->frequency * dt) {
                state->direction = normalize(make_float3(
                    (((float)rand() / RAND_MAX) * 2.0f - 1.0f),
                    (((float)rand() / RAND_MAX) * 2.0f - 1.0f),
                    (((float)rand() / RAND_MAX) * 2.0f - 1.0f)
                ));
            }
            accel = state->direction * g;
            break;
        }
        
        case MANEUVER_STEP_TURN: {
            // Periodic hard turns
            float period = 1.0f / state->frequency;
            int currentStep = (int)(state->time / period);
            if (currentStep > state->stepCount) {
                state->stepCount = currentStep;
                // Random direction for new turn
                float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                state->direction = make_float3(cosf(angle), 0.0f, sinf(angle));
            }
            accel = state->direction * g;
            break;
        }
        
        case MANEUVER_JINK: {
            // Rapid random jinking
            if (((float)rand() / RAND_MAX) < 5.0f * dt) {  // 5 jinks per second avg
                float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
                state->direction = make_float3(cosf(angle), (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.3f, sinf(angle));
                state->direction = normalize(state->direction);
            }
            accel = state->direction * g * (0.5f + 0.5f * ((float)rand() / RAND_MAX));
            break;
        }
        
        case MANEUVER_ADAPTIVE: {
            // If we have interceptor info, evade it
            if (state->hasInterceptorInfo) {
                float3 toInterceptor = state->observedInterceptorPos - position;
                float3 relVel = state->observedInterceptorVel - velocity;
                
                // Evade perpendicular to line of sight
                float3 los = normalize(toInterceptor);
                float3 evadeDir = normalize(cross(los, make_float3(0.0f, 1.0f, 0.0f)));
                
                // Check which side to evade
                if (dot(relVel, evadeDir) > 0.0f) {
                    evadeDir = evadeDir * -1.0f;  // Go the other way
                }
                
                accel = evadeDir * g;
            } else {
                // Default to random walk when no info
                if (((float)rand() / RAND_MAX) < state->frequency * dt) {
                    state->direction = normalize(make_float3(
                        (((float)rand() / RAND_MAX) * 2.0f - 1.0f),
                        (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.5f,
                        (((float)rand() / RAND_MAX) * 2.0f - 1.0f)
                    ));
                }
                accel = state->direction * (g * 0.5f);
            }
            break;
        }
        
        default:
            break;
    }
    
    // Clamp to max maneuver capability
    float accelMag = length(accel);
    float maxAccel = g;
    if (accelMag > maxAccel) {
        accel = accel * (maxAccel / accelMag);
    }
    
    return accel;
}

// ============================================================================
// SCENARIO GENERATOR
// ============================================================================

struct Scenario {
    // Threat configuration
    struct ThreatConfig {
        float3 launchPosition;
        float3 targetPosition;
        float launchSpeed;
        float launchAngle;
        ManeuverState maneuverState;
        TargetType targetType;  // Warhead, decoy, etc.
    } threats[100];
    int numThreats;
    
    // Defense configuration
    float3 defensePosition;
    int numInterceptors;
    
    // Timing
    float* threatLaunchTimes;
    float scenarioDuration;
    
    // Objective
    float3 protectedArea;
    float protectedRadius;
};

// Generate scenario based on curriculum difficulty
__host__ inline Scenario generateScenario(
    CurriculumDifficulty* difficulty,
    curandState* randState
) {
    Scenario scenario;
    memset(&scenario, 0, sizeof(Scenario));
    
    scenario.defensePosition = make_float3(8000.0f, 0.0f, 0.0f);
    scenario.protectedArea = scenario.defensePosition;
    scenario.protectedRadius = 2000.0f;
    scenario.numInterceptors = difficulty->availableInterceptors;
    
    // Generate threats
    int numWarheads = difficulty->maxThreats;
    int numDecoys = difficulty->maxDecoys;
    scenario.numThreats = numWarheads + numDecoys;
    
    scenario.threatLaunchTimes = (float*)malloc(scenario.numThreats * sizeof(float));
    
    float baseRange = difficulty->engagementRange;
    
    for (int i = 0; i < scenario.numThreats; i++) {
        bool isDecoy = (i >= numWarheads);
        
        // Launch position (enemy side)
        float angle = (((float)rand() / RAND_MAX) - 0.5f) * M_PI * 0.5f;
        float range = baseRange + ((float)rand() / RAND_MAX) * 5000.0f;
        scenario.threats[i].launchPosition = make_float3(
            -8000.0f + (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 1000.0f,
            500.0f + ((float)rand() / RAND_MAX) * 500.0f,
            (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 2000.0f
        );
        
        // Target (in protected area)
        float targetAngle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float targetRadius = ((float)rand() / RAND_MAX) * scenario.protectedRadius * 0.8f;
        scenario.threats[i].targetPosition = make_float3(
            scenario.protectedArea.x + targetRadius * cosf(targetAngle),
            0.0f,
            scenario.protectedArea.z + targetRadius * sinf(targetAngle)
        );
        
        // Launch parameters
        scenario.threats[i].launchSpeed = 400.0f + ((float)rand() / RAND_MAX) * 400.0f;
        scenario.threats[i].launchSpeed *= difficulty->targetSpeedMultiplier;
        scenario.threats[i].launchAngle = 0.4f + ((float)rand() / RAND_MAX) * 0.6f;
        
        // Target type
        if (isDecoy) {
            float r = ((float)rand() / RAND_MAX);
            if (r < 0.6f) {
                scenario.threats[i].targetType = TARGET_DECOY_BALLOON;
            } else {
                scenario.threats[i].targetType = TARGET_DECOY_CONE;
            }
        } else {
            float r = ((float)rand() / RAND_MAX);
            if (r < 0.7f) {
                scenario.threats[i].targetType = TARGET_WARHEAD;
            } else {
                scenario.threats[i].targetType = TARGET_HGV;
            }
        }
        
        // Maneuver configuration
        ManeuverState& ms = scenario.threats[i].maneuverState;
        ms.amplitude = difficulty->maxTargetManeuver;
        ms.frequency = difficulty->maneuverFrequency;
        ms.phase = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        ms.time = 0.0f;
        ms.stepCount = 0;
        ms.direction = make_float3(0.0f, 0.0f, 0.0f);
        ms.hasInterceptorInfo = false;
        
        // Choose maneuver type based on difficulty
        float r = ((float)rand() / RAND_MAX);
        if (difficulty->maneuverRandomness < 0.2f) {
            ms.type = MANEUVER_STRAIGHT;
        } else if (difficulty->maneuverRandomness < 0.4f) {
            ms.type = (r < 0.5f) ? MANEUVER_WEAVE_SINE : MANEUVER_WEAVE_SQUARE;
        } else if (difficulty->maneuverRandomness < 0.6f) {
            int type = (int)(r * 4.0f);
            ms.type = (ManeuverType)(MANEUVER_WEAVE_SINE + type);
        } else if (difficulty->maneuverRandomness < 0.8f) {
            ms.type = (r < 0.5f) ? MANEUVER_RANDOM_WALK : MANEUVER_JINK;
        } else {
            ms.type = MANEUVER_ADAPTIVE;
        }
        
        // Stagger launch times
        scenario.threatLaunchTimes[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    scenario.scenarioDuration = 60.0f;
    
    return scenario;
}

// ============================================================================
// CURRICULUM ADVANCEMENT LOGIC
// ============================================================================

__host__ inline void updateCurriculumState(
    CurriculumState* state,
    float episodeSuccessRate,
    float predictionError,
    float discriminationAccuracy
) {
    // Track metrics
    state->recentSuccessRates[state->recentIndex] = episodeSuccessRate;
    state->recentIndex = (state->recentIndex + 1) % 100;
    state->episodesInPhase++;
    state->totalEpisodes++;
    
    // Compute moving average
    float sum = 0.0f;
    int count = fminf(100, state->episodesInPhase);
    for (int i = 0; i < count; i++) {
        sum += state->recentSuccessRates[i];
    }
    state->movingAvgSuccessRate = sum / count;
    
    // Update best
    if (state->movingAvgSuccessRate > state->bestSuccessRate) {
        state->bestSuccessRate = state->movingAvgSuccessRate;
    }
    
    // Track phase-specific metrics
    state->avgPredictionError = 0.9f * state->avgPredictionError + 0.1f * predictionError;
    state->avgDiscriminationAccuracy = 0.9f * state->avgDiscriminationAccuracy + 0.1f * discriminationAccuracy;
    
    // Check for advancement
    if (state->episodesInPhase >= state->minEpisodesBeforeAdvance &&
        state->movingAvgSuccessRate >= state->advancementThreshold) {
        
        // Advance to next phase
        if (state->currentPhase < PHASE_FULL_COMPLEXITY) {
            state->currentPhase = (CurriculumPhase)(state->currentPhase + 1);
            state->episodesInPhase = 0;
            state->bestSuccessRate = 0.0f;
            
            // Update difficulty
            switch (state->currentPhase) {
                case PHASE_BASIC_PN:
                    state->difficulty = getPhaseBasicPN();
                    state->advancementThreshold = 0.9f;  // Need 90% to advance
                    break;
                case PHASE_MANEUVERING:
                    state->difficulty = getPhaseManeuvering();
                    state->advancementThreshold = 0.8f;
                    break;
                case PHASE_MULTI_AGENT:
                    state->difficulty = getPhaseMultiAgent();
                    state->advancementThreshold = 0.7f;
                    break;
                case PHASE_ADVERSARIAL:
                    state->difficulty = getPhaseAdversarial();
                    state->advancementThreshold = 0.6f;
                    break;
                case PHASE_FULL_COMPLEXITY:
                    state->difficulty = getPhaseFullComplexity();
                    state->advancementThreshold = 0.5f;  // Final phase, just maintain
                    break;
            }
            
            printf("Advanced to curriculum phase %d! Avg success: %.2f%%\n",
                   state->currentPhase, state->movingAvgSuccessRate * 100.0f);
        }
    }
}

// ============================================================================
// REWARD SHAPING FOR CURRICULUM
// ============================================================================

// Information gain reward for discrimination
__device__ inline float computeInformationGainReward(
    float* priorProbabilities,    // P(target type) before measurement
    float* posteriorProbabilities, // P(target type) after measurement
    int numTypes
) {
    // KL divergence measures information gain
    float klDiv = 0.0f;
    for (int i = 0; i < numTypes; i++) {
        if (posteriorProbabilities[i] > 1e-6f && priorProbabilities[i] > 1e-6f) {
            klDiv += posteriorProbabilities[i] * 
                     logf(posteriorProbabilities[i] / priorProbabilities[i]);
        }
    }
    
    return klDiv * 10.0f;  // Scale for reward
}

// Entropy reduction reward (more certain = better)
__device__ inline float computeEntropyReductionReward(
    float* probabilities,
    int numTypes
) {
    float entropy = 0.0f;
    for (int i = 0; i < numTypes; i++) {
        if (probabilities[i] > 1e-6f) {
            entropy -= probabilities[i] * logf(probabilities[i]);
        }
    }
    
    // Max entropy for uniform is log(numTypes)
    float maxEntropy = logf((float)numTypes);
    float reduction = maxEntropy - entropy;
    
    return reduction * 5.0f;  // Reward for certainty
}

// Curriculum-phase-specific reward adjustments
__device__ inline float adjustRewardForPhase(
    float baseReward,
    CurriculumPhase phase,
    bool wasWarhead,
    bool correctClassification,
    float timeToIntercept,
    float predictionAccuracy
) {
    float adjustedReward = baseReward;
    
    switch (phase) {
        case PHASE_BASIC_PN:
            // Focus on interception fundamentals
            adjustedReward += predictionAccuracy * 10.0f;  // Bonus for good prediction
            break;
            
        case PHASE_MANEUVERING:
            // Emphasize prediction under maneuvers
            adjustedReward += predictionAccuracy * 20.0f;
            adjustedReward -= (1.0f - predictionAccuracy) * 10.0f;  // Penalty for poor prediction
            break;
            
        case PHASE_MULTI_AGENT:
            // Emphasize discrimination
            if (correctClassification) {
                adjustedReward += 30.0f;
            } else {
                adjustedReward -= 50.0f;  // Heavy penalty for misclassification
            }
            // Bonus for ignoring decoys
            if (!wasWarhead && !correctClassification) {
                adjustedReward -= 100.0f;  // Wasted interceptor on decoy
            }
            break;
            
        case PHASE_ADVERSARIAL:
        case PHASE_FULL_COMPLEXITY:
            // All factors matter
            adjustedReward += predictionAccuracy * 15.0f;
            if (correctClassification) {
                adjustedReward += 20.0f;
            } else {
                adjustedReward -= 40.0f;
            }
            // Time efficiency bonus
            adjustedReward += (20.0f - timeToIntercept) * 2.0f;
            break;
    }
    
    return adjustedReward;
}

// ============================================================================
// ADVERSARIAL SELF-PLAY
// ============================================================================

// Agent that controls the threat (for co-evolution)
struct ThreatAgent {
    // Simple policy network for evasion
    float weights1[12 * 64];  // Input: relative geometry
    float bias1[64];
    float weights2[64 * 3];   // Output: maneuver direction
    float bias2[3];
};

// Threat agent forward pass (for adaptive maneuvers)
__device__ inline float3 threatAgentPolicy(
    ThreatAgent* agent,
    float3 myPos,
    float3 myVel,
    float3 interceptorPos,
    float3 interceptorVel
) {
    // Input features
    float3 relPos = interceptorPos - myPos;
    float3 relVel = interceptorVel - myVel;
    
    float input[12] = {
        relPos.x / 5000.0f, relPos.y / 5000.0f, relPos.z / 5000.0f,
        relVel.x / 1000.0f, relVel.y / 1000.0f, relVel.z / 1000.0f,
        myVel.x / 1000.0f, myVel.y / 1000.0f, myVel.z / 1000.0f,
        length(relPos) / 10000.0f,
        dot(normalize(relVel), normalize(relPos)),
        atan2f(relPos.z, relPos.x) / M_PI
    };
    
    // Hidden layer
    float hidden[64];
    for (int i = 0; i < 64; i++) {
        float sum = agent->bias1[i];
        for (int j = 0; j < 12; j++) {
            sum += input[j] * agent->weights1[j * 64 + i];
        }
        hidden[i] = tanhf(sum);
    }
    
    // Output layer
    float output[3];
    for (int i = 0; i < 3; i++) {
        float sum = agent->bias2[i];
        for (int j = 0; j < 64; j++) {
            sum += hidden[j] * agent->weights2[j * 3 + i];
        }
        output[i] = tanhf(sum);
    }
    
    // Scale to maneuver direction
    float3 maneuverDir = normalize(make_float3(output[0], output[1] * 0.5f, output[2]));
    
    return maneuverDir;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

__host__ inline void initCurriculumState(CurriculumState* state) {
    memset(state, 0, sizeof(CurriculumState));
    
    state->currentPhase = PHASE_BASIC_PN;
    state->difficulty = getPhaseBasicPN();
    state->advancementThreshold = 0.9f;
    state->minEpisodesBeforeAdvance = 100;
    
    for (int i = 0; i < 100; i++) {
        state->recentSuccessRates[i] = 0.0f;
    }
}

__host__ inline void initThreatAgent(ThreatAgent* agent) {
    // Xavier initialization
    float scale = sqrtf(2.0f / (12 + 64));
    for (int i = 0; i < 12 * 64; i++) {
        agent->weights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    scale = sqrtf(2.0f / (64 + 3));
    for (int i = 0; i < 64 * 3; i++) {
        agent->weights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    memset(agent->bias1, 0, sizeof(agent->bias1));
    memset(agent->bias2, 0, sizeof(agent->bias2));
}

#endif // CURRICULUM_LEARNING_CUH
