#ifndef DEFENSE_GRADE_SIMULATION_CUH
#define DEFENSE_GRADE_SIMULATION_CUH

#include "../core/config.h"
#include "sixdof_dynamics.cuh"
#include "sensor_simulation.cuh"
#include "physnode_pinn.cuh"
#include "hierarchical_rl.cuh"
#include "curriculum_learning.cuh"

enum SimulationMode {
    SIM_MODE_LEGACY = 0,
    SIM_MODE_DEFENSE_GRADE = 1,
    SIM_MODE_HYBRID = 2
};

enum CoordinateSystem {
    COORDS_LOCAL_NED = 0,
    COORDS_ECEF = 1,
    COORDS_ECI = 2
};

struct DefenseGradeConfig {
    SimulationMode mode;
    CoordinateSystem coordSystem;
    
    bool use6DOF;
    bool useQuaternions;
    bool useJ2Gravity;
    bool useVariableMass;
    bool useAeroDatabase;
    
    bool useRCSModel;
    bool useMicroDoppler;
    bool useDiscrimination;
    float sensorNoiseLevel;
    
    bool usePhysNODE;
    bool useHierarchicalRL;
    bool useCurriculumLearning;
    int physNodeMCDropoutSamples;
    
    int maxDefenseGradeMissiles;
    float defenseGradeRadius;
};

__host__ inline DefenseGradeConfig getDefaultConfig() {
    DefenseGradeConfig config;
    config.mode = SIM_MODE_DEFENSE_GRADE;
    config.coordSystem = COORDS_LOCAL_NED;
    
    config.use6DOF = true;
    config.useQuaternions = true;
    config.useJ2Gravity = true;
    config.useVariableMass = true;
    config.useAeroDatabase = true;
    
    config.useRCSModel = true;
    config.useMicroDoppler = true;
    config.useDiscrimination = true;
    config.sensorNoiseLevel = 0.1f;
    
    config.usePhysNODE = true;
    config.useHierarchicalRL = true;
    config.useCurriculumLearning = true;
    config.physNodeMCDropoutSamples = MC_DROPOUT_SAMPLES;
    
    config.maxDefenseGradeMissiles = 1000;
    config.defenseGradeRadius = 15000.0f;
    
    return config;
}

struct DefenseGradeState {
    MissileExtended* d_extendedMissiles;
    int numExtendedMissiles;
    int maxExtendedMissiles;
    
    RadarSignature* d_signatures;
    DiscriminationFeatures* d_features;
    float* d_classificationProbs;
    
    PhysNodeNetwork* d_physNode;
    float* d_trajectoryBuffer;
    TrajectoryPrediction* d_predictions;
    
    StrategistNetwork* d_strategist;
    PilotNetwork* d_pilots;
    EngagementState* d_engagements;
    
    CurriculumState curriculumState;
    ThreatAgent* threatAgents;
    int numThreatAgents;
    
    DefenseGradeConfig config;
    
    curandState* d_randStates;
};

// Forward declarations
__global__ void updateDefenseGradeKernel(
    Missile* missiles,
    MissileExtended* extended,
    int* extendedIndices,      // maps missile index -> extended index (-1 if none)
    int numMissiles,
    DefenseGradeConfig config,
    PhysNodeNetwork* physNode,
    StrategistNetwork* strategist,
    PilotNetwork* pilots,
    float dt,
    curandState* randStates
);

// Helper: convert Missile to 6-DOF state
__device__ inline void missileToState6DOF(const Missile& m, State6DOF& s) {
    s.position = m.position;
    s.velocity = m.velocity;
    // assume initial quaternion identity for simplicity (real version would use stored attitude)
    s.attitude = {1.0f, 0.0f, 0.0f, 0.0f};
    s.angularVelocity = make_float3(0.0f, 0.0f, 0.0f);
    s.massProps.mass = m.mass;
    s.massProps.propellantMass = m.fuel;
    s.massProps.burnRate = 0.0f; // enemy already done, interceptors use constant thrust later
    s.altitude = m.position.y;
    s.machNumber = 0.0f;
}

// Helper: write back 6-DOF to Missile
__device__ inline void state6DOFToMissile(const State6DOF& s, Missile& m) {
    m.position = s.position;
    m.velocity = s.velocity;
    m.mass = s.massProps.mass;
    m.fuel = s.massProps.propellantMass;
    // m.acceleration can be computed from forces if needed, set elsewhere
}

// The main defense‑grade kernel
__global__ void updateDefenseGradeKernel(
    Missile* missiles,
    MissileExtended* extended,
    int* extendedIndices,
    int numMissiles,
    DefenseGradeConfig config,
    PhysNodeNetwork* physNode,
    StrategistNetwork* strategist,
    PilotNetwork* pilots,
    float dt,
    curandState* randStates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMissiles) return;
    Missile& m = missiles[idx];
    if (!m.active) return;
    int extIdx = extendedIndices[idx];
    if (extIdx < 0) return; // no extended state => skip (shouldn't happen in defense-grade mode)

    MissileExtended& ext = extended[extIdx];
    curandState& rs = randStates[idx];

    // ========== 6-DOF DYNAMICS ==========
    if (config.use6DOF) {
        State6DOF state;
        missileToState6DOF(m, state);
        // Use sensible mass properties (the extended missile already holds them)
        // We copy from extended
        
        state.massProps.mass = ext.massProps.totalMass;
        state.massProps.propellantMass = ext.massProps.fuelMass;
        state.massProps.Ixx = ext.massProps.inertiaMatrix.x;
        state.massProps.Iyy = ext.massProps.inertiaMatrix.y;
        state.massProps.Izz = ext.massProps.inertiaMatrix.z;

        // Update altitude and mach for aero lookup
        state.altitude = m.position.y;
        // Thrust and control moment from guidance
        float3 thrustBody = make_float3(0,0,0);
        float3 controlMoment = make_float3(0,0,0);
        if (m.type == INTERCEPTOR_MISSILE && m.fuel > 0.0f) {
            // Use HRL pilot command if available
            thrustBody = ext.hrlState.guidanceCommand;
            // For simplicity, set thrust along body x (forward)
            // in a full implementation, align with body frame.
        }
        // RK4 integration over SIXDOF_SUBSTEPS
        float subDt = dt / SIXDOF_SUBSTEPS;
        for (int sub = 0; sub < SIXDOF_SUBSTEPS; sub++) {
            integrate6DOF_RK4(&state, thrustBody, controlMoment,
                              REFERENCE_AREA_MISSILE, 4.0f, subDt);
        }
        // Write back
        state6DOFToMissile(state, m);
        
        ext.massProps.totalMass = state.massProps.mass;
        ext.massProps.fuelMass = state.massProps.propellantMass;
        ext.massProps.inertiaMatrix.x = state.massProps.Ixx;
        ext.massProps.inertiaMatrix.y = state.massProps.Iyy;
        ext.massProps.inertiaMatrix.z = state.massProps.Izz;

        // Store additional derived values for logging
        m.machNumber = state.machNumber;
        m.dynamicPressure = state.dynamicPressure;
    }

    // ========== PhysNODE PREDICTION (Enemy only) ==========
    if (config.usePhysNODE && m.type == ENEMY_MISSILE) {
        // Build observation sequence from history (stored in ext, not shown here)
        // For a first working version, we just do a one-step prediction using current state.
        float obs[PHYSNODE_SEQUENCE_LEN * PHYSNODE_INPUT_DIM];
        // In practice we would maintain a circular buffer in MissileExtended.
        // Here we simply fill with repeated current state.
        for (int t = 0; t < PHYSNODE_SEQUENCE_LEN; t++) {
            // fill with normalized position/velocity etc.
            obs[t*PHYSNODE_INPUT_DIM + 0] = m.position.x / 10000.0f;
            obs[t*PHYSNODE_INPUT_DIM + 1] = m.position.y / 10000.0f;
            obs[t*PHYSNODE_INPUT_DIM + 2] = m.position.z / 10000.0f;
            obs[t*PHYSNODE_INPUT_DIM + 3] = m.velocity.x / 1000.0f;
            obs[t*PHYSNODE_INPUT_DIM + 4] = m.velocity.y / 1000.0f;
            obs[t*PHYSNODE_INPUT_DIM + 5] = m.velocity.z / 1000.0f;
            // ... other features set to zero for now
        }
        PhysNodePrediction pred = physNodePredict(physNode, obs, m.position, m.velocity, 2.0f, 0.1f);
        ext.prediction.predictedPositions[0] = pred.position;
        ext.prediction.predictedVelocities[0] = pred.velocity;
        ext.prediction.confidence = pred.confidence;
    }

    // ========== HRL GUIDANCE (Interceptors) ==========
    if (config.useHierarchicalRL && m.type == INTERCEPTOR_MISSILE) {
        // Build engagement state from the assigned target
        int targetId = m.targetMissileId;
        if (targetId >= 0 && targetId < numMissiles) {
            Missile& target = missiles[targetId];
            EngagementState eng;
            eng.range = length(target.position - m.position);
            eng.rangeRate = -dot(target.velocity - m.velocity, normalize(target.position - m.position));
            eng.lineOfSight = normalize(target.position - m.position);
            float3 relVel = target.velocity - m.velocity;
            eng.lineOfSightRate = (relVel - eng.lineOfSight*dot(relVel, eng.lineOfSight)) / eng.range;
            eng.targetVelocity = target.velocity;
            eng.targetAcceleration = target.acceleration;
            eng.interceptorVelocity = m.velocity;
            eng.fuel = m.fuel;
            eng.machNumber = m.machNumber;
            eng.timeToGo = eng.range / fmaxf(eng.rangeRate, 1.0f);
            eng.zeroEffortMiss = length(cross(relVel, eng.lineOfSight*eng.range)) / fmaxf(length(relVel), 1.0f);
            eng.maxAcceleration = MAX_MANEUVER_G * GRAVITY;
            eng.inTerminalPhase = (eng.range < 500.0f);

            // Get pilot action
            float pilotState[PILOT_STATE_DIM];
            encodeEngagementState(&eng, pilotState);
            float actionMean[PILOT_ACTION_DIM], actionLogStd[PILOT_ACTION_DIM];
            float sampledAction[PILOT_ACTION_DIM];
            pilotPolicyForward(&pilots[extIdx], pilotState, actionMean, actionLogStd, &rs, sampledAction);
            PilotAction pilotAction;
            pilotAction.pitchCommand = sampledAction[0];
            pilotAction.yawCommand = sampledAction[1];
            pilotAction.rollCommand = sampledAction[2];
            pilotAction.throttleCommand = (sampledAction[3]+1.0f)*0.5f;

            // Convert to acceleration command (simplified)
            float scale = MAX_MANEUVER_G * GRAVITY;
            pilotAction.commandedAcceleration = make_float3(
                pilotAction.yawCommand * scale,
                pilotAction.pitchCommand * scale,
                pilotAction.throttleCommand * 0.5f * scale  // axial thrust component
            );

            // MPC refinement (call solvePilotMPC using engagement state and action)
            MPCConstraints mpc = {MAX_MANEUVER_G, 10.0f, MAX_AOA_RAD, 25.0f*M_PI/180.0f, 100.0f*M_PI/180.0f, 100.0f};
            MPCSolution mpcSol = solvePilotMPC(&eng, &pilotAction, &mpc, dt);
            ext.hrlState.guidanceCommand = mpcSol.commandedAccel;
        } else {
            // No target, just coast
            ext.hrlState.guidanceCommand = make_float3(0,0,0);
        }
    }

    // ========== Ground collision and out-of-bounds ==========
    if (m.position.y <= GROUND_LEVEL) {
        m.position.y = GROUND_LEVEL;
        m.velocity = make_float3(0,0,0);
        m.active = 0;
        m.hit = 1;
        // Set termination state based on type (handled elsewhere)
    }
}

#endif // DEFENSE_GRADE_SIMULATION_CUH
