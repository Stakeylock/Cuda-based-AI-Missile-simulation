#ifndef PHYSX_ENGINE_CUH
#define PHYSX_ENGINE_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"

// ============================================================================
// NVIDIA PHYSX-STYLE PHYSICS ENGINE FOR CUDA
// ============================================================================
// This implements a PhysX-compatible physics simulation directly on GPU
// for accurate missile trajectory simulation

// ============================================================================
// ATMOSPHERIC MODEL (ISA - International Standard Atmosphere)
// ============================================================================
__host__ __device__ inline float getAirDensity(float altitude) {
    // ISA model for air density vs altitude
    if (altitude < 0.0f) altitude = 0.0f;
    if (altitude > 80000.0f) altitude = 80000.0f;
    
    float temperature = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude;
    if (temperature < 216.65f) temperature = 216.65f;  // Stratosphere limit
    
    float pressure = SEA_LEVEL_PRESSURE * 
        powf(temperature / SEA_LEVEL_TEMP, 
             GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * TEMP_LAPSE_RATE));
    
    return (pressure * MOLAR_MASS_AIR) / (GAS_CONSTANT * temperature);
}

__host__ __device__ inline float getTemperature(float altitude) {
    if (altitude < 0.0f) altitude = 0.0f;
    float temp = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude;
    return fmaxf(temp, 216.65f);
}

__host__ __device__ inline float getSpeedOfSound(float altitude) {
    float temp = getTemperature(altitude);
    // Speed of sound = sqrt(gamma * R * T / M)
    // For air: gamma = 1.4, R = 8.314, M = 0.029
    return sqrtf(1.4f * 8.314f * temp / 0.029f);
}

__host__ __device__ inline float getMachNumber(float speed, float altitude) {
    float soundSpeed = getSpeedOfSound(altitude);
    return speed / soundSpeed;
}

// ============================================================================
// DRAG MODEL (Subsonic, Transonic, Supersonic)
// ============================================================================
__host__ __device__ inline float getDragCoefficient(float mach, float baseCD) {
    // Wave drag increases dramatically near Mach 1
    if (mach < 0.8f) {
        // Subsonic - relatively constant
        return baseCD;
    } else if (mach < 1.2f) {
        // Transonic - peak drag
        float t = (mach - 0.8f) / 0.4f;
        return baseCD * (1.0f + 0.8f * sinf(t * M_PI));
    } else if (mach < 3.0f) {
        // Supersonic - decreasing drag
        return baseCD * (1.0f + 0.4f / (mach - 0.5f));
    } else {
        // Hypersonic
        return baseCD * 0.5f;
    }
}

// ============================================================================
// LIFT MODEL
// ============================================================================
__host__ __device__ inline float getLiftCoefficient(float angleOfAttack, float mach) {
    // Simplified lift coefficient based on angle of attack
    float clAlpha = 2.0f * M_PI;  // Thin airfoil theory
    
    // Mach number correction (Prandtl-Glauert)
    if (mach < 1.0f && mach > 0.1f) {
        clAlpha /= sqrtf(1.0f - mach * mach);
    } else if (mach >= 1.0f && mach < 2.0f) {
        clAlpha /= sqrtf(mach * mach - 1.0f);
    }
    
    // Stall model
    float stallAngle = 15.0f * M_PI / 180.0f;
    if (fabsf(angleOfAttack) > stallAngle) {
        float stallFactor = 1.0f - (fabsf(angleOfAttack) - stallAngle) / (M_PI / 4.0f);
        clAlpha *= fmaxf(stallFactor, 0.2f);
    }
    
    return clAlpha * angleOfAttack;
}

// ============================================================================
// PHYSX-STYLE RIGID BODY DYNAMICS
// ============================================================================
__device__ inline void physxComputeForces(
    Missile* m,
    float3 targetDir,
    float thrustMag,
    float dt
) {
    // Get atmospheric conditions
    m->altitude = m->position.y;
    m->airDensity = getAirDensity(m->altitude);
    m->temperature = getTemperature(m->altitude);
    
    float speed = length(m->velocity);
    m->machNumber = getMachNumber(speed, m->altitude);
    
    // Dynamic pressure: q = 0.5 * rho * V^2
    m->dynamicPressure = 0.5f * m->airDensity * speed * speed;
    
    // Track max velocity
    if (speed > m->maxVelocity) {
        m->maxVelocity = speed;
    }
    
    // Initialize forces
    m->force = make_float3(0.0f, 0.0f, 0.0f);
    m->torque = make_float3(0.0f, 0.0f, 0.0f);
    
    // 1. GRAVITY
    float3 gravity = make_float3(0.0f, -GRAVITY * m->mass, 0.0f);
    m->force = m->force + gravity;
    
    // 2. THRUST (if fuel available)
    if (m->fuel > 0.0f && thrustMag > 0.0f) {
        float3 thrustDir = normalize(m->velocity);
        if (length(m->velocity) < 1.0f) {
            thrustDir = targetDir;
        }
        
        // Apply steering towards target
        float3 steerDir = normalize(targetDir - thrustDir);
        float steerAmount = fminf(MAX_TURN_RATE * dt, 1.0f);
        thrustDir = normalize(thrustDir + steerDir * steerAmount);
        
        float3 thrust = thrustDir * thrustMag;
        m->force = m->force + thrust;
        
        // Update average thrust
        m->averageThrust = (m->averageThrust * m->lifetime + thrustMag * dt) / (m->lifetime + dt);
        
        // Fuel consumption based on thrust
        float fuelRate = (m->type == INTERCEPTOR_MISSILE) ? 3.0f : 2.0f;
        m->fuel -= fuelRate * dt;
        m->fuelConsumed += fuelRate * dt;
    }
    
    // 3. AERODYNAMIC DRAG
    if (speed > 0.1f) {
        float CD = getDragCoefficient(m->machNumber, m->dragCoefficient);
        float dragMag = m->dynamicPressure * CD * m->referenceArea;
        float3 dragDir = normalize(m->velocity) * (-1.0f);
        float3 drag = dragDir * dragMag;
        m->force = m->force + drag;
    }
    
    // 4. LIFT (for maneuvering)
    if (speed > 10.0f) {
        // Calculate angle of attack
        float3 velDir = normalize(m->velocity);
        float3 bodyAxis = normalize(targetDir);
        float dotProd = fminf(fmaxf(dot(velDir, bodyAxis), -1.0f), 1.0f);
        float angleOfAttack = acosf(dotProd);
        
        float CL = getLiftCoefficient(angleOfAttack, m->machNumber);
        float liftMag = m->dynamicPressure * CL * m->referenceArea;
        
        // Lift perpendicular to velocity
        float3 liftDir = normalize(bodyAxis - velDir * dot(bodyAxis, velDir));
        if (length(liftDir) < 0.001f) {
            liftDir = make_float3(0.0f, 1.0f, 0.0f);
        }
        float3 lift = liftDir * liftMag;
        m->force = m->force + lift;
    }
    
    // 5. MAGNUS EFFECT (spin stabilization)
    if (speed > 10.0f && length(m->angularVelocity) > 0.01f) {
        float3 magnusDir = normalize(cross(m->angularVelocity, m->velocity));
        float magnusMag = MAGNUS_COEFFICIENT * m->airDensity * speed * 
                          length(m->angularVelocity) * m->referenceArea;
        float3 magnus = magnusDir * magnusMag;
        m->force = m->force + magnus;
    }
}

// ============================================================================
// PHYSX INTEGRATION (Semi-implicit Euler with substeps)
// ============================================================================
__device__ inline void physxIntegrate(Missile* m, float dt) {
    // Store previous state for statistics
    m->prevPosition = m->position;
    m->prevVelocity = m->velocity;
    
    // Calculate acceleration
    m->acceleration = m->force / m->mass;
    
    // Track max acceleration
    float accelMag = length(m->acceleration);
    if (accelMag > m->maxAcceleration) {
        m->maxAcceleration = accelMag;
    }
    
    // Semi-implicit Euler integration
    m->velocity = m->velocity + m->acceleration * dt;
    
    // Apply linear damping (air resistance at molecular level)
    m->velocity = m->velocity * (1.0f - PHYSX_LINEAR_DAMPING * dt);
    
    // Clamp velocity to prevent numerical instability
    float maxSpeed = 3000.0f;  // ~Mach 9
    float speed = length(m->velocity);
    if (speed > maxSpeed) {
        m->velocity = normalize(m->velocity) * maxSpeed;
    }
    
    // Update position
    m->position = m->position + m->velocity * dt;
    
    // Update angular velocity with damping
    m->angularVelocity = m->angularVelocity * (1.0f - PHYSX_ANGULAR_DAMPING * dt);
    
    // Update orientation based on angular velocity
    m->orientation = m->orientation + m->angularVelocity * dt;
    
    // Track distance traveled
    float3 displacement = m->position - m->prevPosition;
    m->distanceTraveled += length(displacement);
    
    // Track max altitude
    if (m->position.y > m->maxAltitude) {
        m->maxAltitude = m->position.y;
    }
    
    // Update lifetime
    m->lifetime += dt;
}

// ============================================================================
// PHYSX SUBSTEP SIMULATION
// ============================================================================
__device__ inline void physxSimulateSubsteps(
    Missile* m,
    float3 targetDir,
    float thrustMag,
    float dt
) {
    float substepDt = dt / (float)PHYSX_SUBSTEPS;
    
    for (int i = 0; i < PHYSX_SUBSTEPS; i++) {
        physxComputeForces(m, targetDir, thrustMag, substepDt);
        physxIntegrate(m, substepDt);
    }
}

// ============================================================================
// BALLISTIC TRAJECTORY PREDICTION (for enemy missiles)
// ============================================================================
__device__ inline float3 physxPredictBallisticImpact(
    float3 startPos,
    float3 startVel,
    float fuel,
    float thrustDuration,
    float3 targetDir,
    float thrustMag,
    float mass
) {
    // Simulate forward to find impact point
    float3 pos = startPos;
    float3 vel = startVel;
    float currentFuel = fuel;
    float dt = 0.1f;
    
    for (int step = 0; step < 1000; step++) {
        // Gravity
        float3 accel = make_float3(0.0f, -GRAVITY, 0.0f);
        
        // Thrust if fuel available
        if (currentFuel > 0.0f) {
            float3 thrustAccel = normalize(targetDir) * (thrustMag / mass);
            accel = accel + thrustAccel;
            currentFuel -= 2.0f * dt;
        }
        
        // Air density at altitude
        float airDens = getAirDensity(pos.y);
        float speed = length(vel);
        
        // Drag
        if (speed > 0.1f) {
            float dragMag = 0.5f * airDens * speed * speed * DRAG_COEFFICIENT * 0.5f;
            float3 dragAccel = normalize(vel) * (-dragMag / mass);
            accel = accel + dragAccel;
        }
        
        // Integration
        vel = vel + accel * dt;
        pos = pos + vel * dt;
        
        // Ground collision
        if (pos.y <= GROUND_LEVEL) {
            return pos;
        }
        
        // Out of bounds
        if (fabsf(pos.x) > WORLD_SIZE * 2 || fabsf(pos.z) > WORLD_SIZE * 2) {
            return pos;
        }
    }
    
    return pos;
}

// ============================================================================
// INTERCEPT POINT CALCULATION (PhysX-enhanced)
// ============================================================================
__device__ inline float3 physxCalculateInterceptPoint(
    float3 missilePos,
    float3 missileVel,
    float3 defensePos,
    float interceptorSpeed,
    float missileFuel,
    float3 missileTarget
) {
    // First, predict where the missile will be
    float3 predictedImpact = physxPredictBallisticImpact(
        missilePos, missileVel, missileFuel, 10.0f,
        normalize(missileTarget - missilePos), THRUST_FORCE, MISSILE_MASS
    );
    
    // Calculate time to reach various points along trajectory
    float3 bestIntercept = missilePos;
    float minTime = 1e10f;
    
    float dt = 0.1f;
    float3 simPos = missilePos;
    float3 simVel = missileVel;
    float simFuel = missileFuel;
    
    for (int step = 0; step < 200; step++) {
        // Simulate missile forward
        float3 accel = make_float3(0.0f, -GRAVITY, 0.0f);
        if (simFuel > 0.0f) {
            float3 thrustDir = normalize(missileTarget - simPos);
            accel = accel + thrustDir * (THRUST_FORCE / MISSILE_MASS);
            simFuel -= 2.0f * dt;
        }
        
        // Drag
        float speed = length(simVel);
        if (speed > 0.1f) {
            float airDens = getAirDensity(simPos.y);
            float dragMag = 0.5f * airDens * speed * speed * DRAG_COEFFICIENT * 0.5f;
            accel = accel - normalize(simVel) * (dragMag / MISSILE_MASS);
        }
        
        simVel = simVel + accel * dt;
        simPos = simPos + simVel * dt;
        
        if (simPos.y <= GROUND_LEVEL) break;
        
        // Time for interceptor to reach this point
        float distToIntercept = length(simPos - defensePos);
        float interceptorTime = distToIntercept / interceptorSpeed;
        float missileTime = step * dt;
        
        // Find point where interceptor can arrive before missile
        if (interceptorTime < missileTime + 2.0f && interceptorTime < minTime) {
            minTime = interceptorTime;
            bestIntercept = simPos;
        }
    }
    
    return bestIntercept;
}

// ============================================================================
// PROPORTIONAL NAVIGATION GUIDANCE
// ============================================================================
__host__ __device__ inline float3 physxProportionalNavigation(
    float3 interceptorPos,
    float3 interceptorVel,
    float3 targetPos,
    float3 targetVel,
    float navGain  // Usually 3-5
) {
    float3 relPos = targetPos - interceptorPos;
    float3 relVel = targetVel - interceptorVel;
    
    float range = length(relPos);
    if (range < 1.0f) range = 1.0f;
    
    // Line of sight rate
    float3 los = relPos / range;
    float closingSpeed = -dot(relVel, los);
    
    // LOS rate vector
    float3 losRate = (relVel - los * dot(relVel, los)) / range;
    
    // Commanded acceleration perpendicular to LOS
    float3 accelCmd = los * closingSpeed * navGain + 
                      cross(los, cross(losRate, los)) * range * navGain;
    
    return accelCmd;
}

// ============================================================================
// COLLISION DETECTION (PhysX-style)
// ============================================================================
__device__ inline bool physxCheckCollision(
    float3 pos1, float radius1,
    float3 pos2, float radius2,
    float3 vel1, float3 vel2,
    float dt,
    float3* collisionPoint
) {
    // Continuous collision detection using swept spheres
    float3 relPos = pos2 - pos1;
    float3 relVel = vel2 - vel1;
    float combRadius = radius1 + radius2;
    
    // Current distance
    float dist = length(relPos);
    if (dist < combRadius) {
        *collisionPoint = (pos1 + pos2) * 0.5f;
        return true;
    }
    
    // Check if approaching
    float closingSpeed = dot(relVel, relPos) / dist;
    if (closingSpeed >= 0.0f) return false;  // Moving apart
    
    // Time to closest approach
    float a = dot(relVel, relVel);
    if (a < 1e-6f) return false;
    
    float b = 2.0f * dot(relPos, relVel);
    float c = dot(relPos, relPos) - combRadius * combRadius;
    
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return false;
    
    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
    if (t < 0.0f || t > dt) return false;
    
    // Collision will occur
    *collisionPoint = pos1 + vel1 * t + (relPos + relVel * t) * (radius1 / combRadius);
    return true;
}

// ============================================================================
// INVERSE BALLISTIC TRAJECTORY SOLVER
// Computes optimal launch angle and velocity to hit a target at (targetX, targetZ)
// Uses iterative Newton-Raphson with drag compensation
// ============================================================================

// Structure to hold inverse trajectory solution
struct InverseTrajectoryResult {
    float launchAngle;      // Elevation angle (radians)
    float launchAzimuth;    // Horizontal direction (radians)
    float launchSpeed;      // Initial velocity magnitude
    float3 initialVelocity; // Full velocity vector
    float flightTime;       // Estimated time to target
    float maxAltitude;      // Estimated peak altitude
    bool valid;             // Whether solution was found
    int iterations;         // Solver iterations used
    float finalError;       // Distance error at solution
};

// Simulate trajectory forward to get impact point
__host__ __device__ inline float3 simulateTrajectoryToGround(
    float3 startPos,
    float3 startVel,
    float mass,
    float dragCoef,
    float refArea,
    float fuel,
    float thrustMag,
    float* outFlightTime,
    float* outMaxAlt
) {
    float3 pos = startPos;
    float3 vel = startVel;
    float currentFuel = fuel;
    float dt = 0.05f;  // 50ms timestep for accuracy
    float flightTime = 0.0f;
    float maxAlt = startPos.y;
    
    for (int step = 0; step < 2000; step++) {
        // Gravity
        float3 accel = make_float3(0.0f, -GRAVITY, 0.0f);
        
        // Thrust phase (first few seconds)
        if (currentFuel > 0.0f && flightTime < 3.0f) {
            float3 thrustDir = normalize(vel);
            if (length(vel) < 1.0f) thrustDir = make_float3(0.0f, 1.0f, 0.0f);
            accel = accel + thrustDir * (thrustMag / mass);
            currentFuel -= 2.0f * dt;
        }
        
        // Atmospheric drag
        float altitude = pos.y;
        float airDens = getAirDensity(altitude);
        float speed = length(vel);
        
        if (speed > 1.0f) {
            float mach = getMachNumber(speed, altitude);
            float CD = getDragCoefficient(mach, dragCoef);
            float dragMag = 0.5f * airDens * speed * speed * CD * refArea;
            float3 dragAccel = normalize(vel) * (-dragMag / mass);
            accel = accel + dragAccel;
        }
        
        // Integration (semi-implicit Euler)
        vel = vel + accel * dt;
        pos = pos + vel * dt;
        flightTime += dt;
        
        if (pos.y > maxAlt) maxAlt = pos.y;
        
        // Ground collision
        if (pos.y <= GROUND_LEVEL) {
            // Interpolate to exact ground impact
            float t_ground = (GROUND_LEVEL - (pos.y - vel.y * dt)) / vel.y;
            pos = pos - vel * (dt - t_ground);
            pos.y = GROUND_LEVEL;
            break;
        }
        
        // Sanity bounds
        if (fabsf(pos.x) > WORLD_SIZE * 3 || fabsf(pos.z) > WORLD_SIZE * 3 || flightTime > 120.0f) {
            break;
        }
    }
    
    if (outFlightTime) *outFlightTime = flightTime;
    if (outMaxAlt) *outMaxAlt = maxAlt;
    
    return pos;
}

// Main inverse trajectory solver
__host__ inline InverseTrajectoryResult solveInverseTrajectory(
    float3 launchPos,
    float3 targetPos,
    float mass,
    float dragCoef,
    float refArea,
    float fuel,
    float thrustMag,
    float minSpeed,
    float maxSpeed
) {
    InverseTrajectoryResult result;
    memset(&result, 0, sizeof(InverseTrajectoryResult));
    result.valid = false;
    
    // Calculate horizontal distance and direction
    float3 flatDir = make_float3(targetPos.x - launchPos.x, 0.0f, targetPos.z - launchPos.z);
    float horizontalDist = sqrtf(flatDir.x * flatDir.x + flatDir.z * flatDir.z);
    
    if (horizontalDist < 100.0f) {
        // Target too close, use default lob
        result.launchAngle = M_PI / 3.0f;
        result.launchSpeed = minSpeed;
        result.launchAzimuth = atan2f(flatDir.z, flatDir.x);
        result.initialVelocity = make_float3(
            result.launchSpeed * cosf(result.launchAngle) * cosf(result.launchAzimuth),
            result.launchSpeed * sinf(result.launchAngle),
            result.launchSpeed * cosf(result.launchAngle) * sinf(result.launchAzimuth)
        );
        result.valid = true;
        return result;
    }
    
    // Azimuth angle (horizontal direction)
    result.launchAzimuth = atan2f(flatDir.z, flatDir.x);
    
    // Binary search + Newton-Raphson for optimal launch parameters
    // We'll search over launch angles from 15° to 75°
    float bestAngle = M_PI / 4.0f;
    float bestSpeed = (minSpeed + maxSpeed) / 2.0f;
    float bestError = 1e10f;
    float bestFlightTime = 0.0f;
    float bestMaxAlt = 0.0f;
    
    // Grid search for initial guess
    for (float angle = 0.26f; angle <= 1.31f; angle += 0.1f) {  // 15° to 75°
        for (float speed = minSpeed; speed <= maxSpeed; speed += 50.0f) {
            float3 vel = make_float3(
                speed * cosf(angle) * cosf(result.launchAzimuth),
                speed * sinf(angle),
                speed * cosf(angle) * sinf(result.launchAzimuth)
            );
            
            float flightTime, maxAlt;
            float3 impact = simulateTrajectoryToGround(
                launchPos, vel, mass, dragCoef, refArea, fuel, thrustMag,
                &flightTime, &maxAlt
            );
            
            float errorX = impact.x - targetPos.x;
            float errorZ = impact.z - targetPos.z;
            float error = sqrtf(errorX * errorX + errorZ * errorZ);
            
            if (error < bestError) {
                bestError = error;
                bestAngle = angle;
                bestSpeed = speed;
                bestFlightTime = flightTime;
                bestMaxAlt = maxAlt;
            }
        }
    }
    
    // Newton-Raphson refinement
    float dAngle = 0.01f;
    float dSpeed = 5.0f;
    
    for (int iter = 0; iter < 20; iter++) {
        result.iterations = iter + 1;
        
        // Compute Jacobian numerically
        float3 vel0 = make_float3(
            bestSpeed * cosf(bestAngle) * cosf(result.launchAzimuth),
            bestSpeed * sinf(bestAngle),
            bestSpeed * cosf(bestAngle) * sinf(result.launchAzimuth)
        );
        
        float3 impact0 = simulateTrajectoryToGround(
            launchPos, vel0, mass, dragCoef, refArea, fuel, thrustMag, NULL, NULL
        );
        
        float error0X = impact0.x - targetPos.x;
        float error0Z = impact0.z - targetPos.z;
        float error0 = sqrtf(error0X * error0X + error0Z * error0Z);
        
        if (error0 < 50.0f) {  // Converged within 50m
            bestError = error0;
            break;
        }
        
        // Partial derivatives w.r.t. angle
        float3 velA = make_float3(
            bestSpeed * cosf(bestAngle + dAngle) * cosf(result.launchAzimuth),
            bestSpeed * sinf(bestAngle + dAngle),
            bestSpeed * cosf(bestAngle + dAngle) * sinf(result.launchAzimuth)
        );
        float3 impactA = simulateTrajectoryToGround(
            launchPos, velA, mass, dragCoef, refArea, fuel, thrustMag, NULL, NULL
        );
        float dImpact_dAngle_X = (impactA.x - impact0.x) / dAngle;
        float dImpact_dAngle_Z = (impactA.z - impact0.z) / dAngle;
        
        // Partial derivatives w.r.t. speed
        float3 velS = make_float3(
            (bestSpeed + dSpeed) * cosf(bestAngle) * cosf(result.launchAzimuth),
            (bestSpeed + dSpeed) * sinf(bestAngle),
            (bestSpeed + dSpeed) * cosf(bestAngle) * sinf(result.launchAzimuth)
        );
        float3 impactS = simulateTrajectoryToGround(
            launchPos, velS, mass, dragCoef, refArea, fuel, thrustMag, NULL, NULL
        );
        float dImpact_dSpeed_X = (impactS.x - impact0.x) / dSpeed;
        float dImpact_dSpeed_Z = (impactS.z - impact0.z) / dSpeed;
        
        // Solve 2x2 linear system: J * [deltaAngle, deltaSpeed]^T = -[errorX, errorZ]^T
        float det = dImpact_dAngle_X * dImpact_dSpeed_Z - dImpact_dAngle_Z * dImpact_dSpeed_X;
        if (fabsf(det) < 1e-6f) break;  // Singular Jacobian
        
        float deltaAngle = (-error0X * dImpact_dSpeed_Z + error0Z * dImpact_dSpeed_X) / det;
        float deltaSpeed = (-error0Z * dImpact_dAngle_X + error0X * dImpact_dAngle_Z) / det;
        
        // Damped update with bounds
        float dampingAngle = 0.5f;
        float dampingSpeed = 0.5f;
        
        bestAngle += dampingAngle * deltaAngle;
        bestSpeed += dampingSpeed * deltaSpeed;
        
        // Clamp to valid ranges
        bestAngle = fmaxf(0.17f, fminf(1.40f, bestAngle));  // 10° to 80°
        bestSpeed = fmaxf(minSpeed, fminf(maxSpeed, bestSpeed));
        
        bestError = error0;
    }
    
    // Final simulation to get accurate flight parameters
    float3 finalVel = make_float3(
        bestSpeed * cosf(bestAngle) * cosf(result.launchAzimuth),
        bestSpeed * sinf(bestAngle),
        bestSpeed * cosf(bestAngle) * sinf(result.launchAzimuth)
    );
    
    float3 finalImpact = simulateTrajectoryToGround(
        launchPos, finalVel, mass, dragCoef, refArea, fuel, thrustMag,
        &bestFlightTime, &bestMaxAlt
    );
    
    float finalErrorX = finalImpact.x - targetPos.x;
    float finalErrorZ = finalImpact.z - targetPos.z;
    bestError = sqrtf(finalErrorX * finalErrorX + finalErrorZ * finalErrorZ);
    
    result.launchAngle = bestAngle;
    result.launchSpeed = bestSpeed;
    result.initialVelocity = finalVel;
    result.flightTime = bestFlightTime;
    result.maxAltitude = bestMaxAlt;
    result.finalError = bestError;
    result.valid = (bestError < 1500.0f);  // Accept if within 1.5km
    
    return result;
}

// Simplified host function for diverse angle generation
__host__ inline InverseTrajectoryResult solveTrajectoryWithRandomAngle(
    float3 launchPos,
    float3 targetPos,
    float mass,
    float dragCoef,
    float refArea,
    float fuel,
    float thrustMag,
    float desiredAngle,  // Force a specific elevation angle
    float minSpeed,
    float maxSpeed
) {
    InverseTrajectoryResult result;
    memset(&result, 0, sizeof(InverseTrajectoryResult));
    
    float3 flatDir = make_float3(targetPos.x - launchPos.x, 0.0f, targetPos.z - launchPos.z);
    float horizontalDist = sqrtf(flatDir.x * flatDir.x + flatDir.z * flatDir.z);
    
    result.launchAzimuth = atan2f(flatDir.z, flatDir.x);
    result.launchAngle = desiredAngle;
    
    // Binary search for speed that hits target with given angle
    float bestSpeed = minSpeed;
    float bestError = 1e10f;
    
    for (float speed = minSpeed; speed <= maxSpeed; speed += 10.0f) {
        float3 vel = make_float3(
            speed * cosf(desiredAngle) * cosf(result.launchAzimuth),
            speed * sinf(desiredAngle),
            speed * cosf(desiredAngle) * sinf(result.launchAzimuth)
        );
        
        float3 impact = simulateTrajectoryToGround(
            launchPos, vel, mass, dragCoef, refArea, fuel, thrustMag, NULL, NULL
        );
        
        float errorX = impact.x - targetPos.x;
        float errorZ = impact.z - targetPos.z;
        float error = sqrtf(errorX * errorX + errorZ * errorZ);
        
        if (error < bestError) {
            bestError = error;
            bestSpeed = speed;
        }
    }
    
    result.launchSpeed = bestSpeed;
    result.initialVelocity = make_float3(
        bestSpeed * cosf(desiredAngle) * cosf(result.launchAzimuth),
        bestSpeed * sinf(desiredAngle),
        bestSpeed * cosf(desiredAngle) * sinf(result.launchAzimuth)
    );
    result.finalError = bestError;
    result.valid = (bestError < 1000.0f);
    
    return result;
}

// ============================================================================
// MODEL PREDICTIVE CONTROL (MPC) FOR INTERCEPTOR GUIDANCE
// Low-level controller that works with high-level RL policy
// ============================================================================

#define MPC_HORIZON 10      // Prediction horizon steps
#define MPC_DT 0.1f         // MPC timestep
#define MPC_MAX_ACCEL 50.0f // Max lateral acceleration

struct MPCState {
    float3 position;
    float3 velocity;
    float fuel;
};

struct MPCControl {
    float3 acceleration;  // Commanded acceleration
    float thrust;         // Thrust magnitude
    float cost;           // Total cost of trajectory
};

// Predict interceptor state forward one step
__host__ __device__ inline MPCState mpcPredictState(
    MPCState state,
    float3 accel,
    float thrust,
    float dt
) {
    MPCState next;
    
    // Apply thrust along velocity direction
    float3 thrustDir = length(state.velocity) > 1.0f ? 
                       normalize(state.velocity) : make_float3(0.0f, 1.0f, 0.0f);
    float3 totalAccel = thrustDir * thrust + accel;
    totalAccel.y -= GRAVITY;  // Add gravity
    
    // Drag
    float speed = length(state.velocity);
    if (speed > 1.0f) {
        float airDens = getAirDensity(state.position.y);
        float dragMag = 0.5f * airDens * speed * speed * DRAG_COEFFICIENT * REFERENCE_AREA_MISSILE / MISSILE_MASS;
        totalAccel = totalAccel - normalize(state.velocity) * dragMag;
    }
    
    next.velocity = state.velocity + totalAccel * dt;
    next.position = state.position + next.velocity * dt;
    next.fuel = state.fuel - (thrust > 0.0f ? 3.0f * dt : 0.0f);
    
    return next;
}

// Compute cost for predicted trajectory reaching target
__host__ __device__ inline float mpcComputeCost(
    float3 predictedPos,
    float3 targetPos,
    float3 accel,
    float distanceWeight,
    float controlWeight,
    float timeWeight,
    float step
) {
    float3 error = predictedPos - targetPos;
    float distCost = distanceWeight * (error.x * error.x + error.y * error.y + error.z * error.z);
    float ctrlCost = controlWeight * (accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
    float timeCost = timeWeight * step;
    
    return distCost + ctrlCost + timeCost;
}

// Solve MPC for optimal control sequence
__host__ inline MPCControl solveMPC(
    float3 interceptorPos,
    float3 interceptorVel,
    float3 targetPos,
    float3 targetVel,
    float fuel,
    float thrustMag
) {
    MPCControl bestControl;
    memset(&bestControl, 0, sizeof(MPCControl));
    bestControl.cost = 1e10f;
    
    // Predict target position at each horizon step
    float3 targetPredictions[MPC_HORIZON];
    float3 tPos = targetPos;
    float3 tVel = targetVel;
    for (int i = 0; i < MPC_HORIZON; i++) {
        tVel.y -= GRAVITY * MPC_DT;
        tPos = tPos + tVel * MPC_DT;
        targetPredictions[i] = tPos;
    }
    
    // Sample candidate accelerations (simplified MPC)
    const int NUM_SAMPLES = 27;  // 3^3 for xyz
    float accelChoices[3] = {-MPC_MAX_ACCEL, 0.0f, MPC_MAX_ACCEL};
    
    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
                float3 testAccel = make_float3(accelChoices[ix], accelChoices[iy], accelChoices[iz]);
                
                // Simulate forward with this control
                MPCState state;
                state.position = interceptorPos;
                state.velocity = interceptorVel;
                state.fuel = fuel;
                
                float totalCost = 0.0f;
                
                for (int step = 0; step < MPC_HORIZON; step++) {
                    state = mpcPredictState(state, testAccel, thrustMag, MPC_DT);
                    
                    float stepCost = mpcComputeCost(
                        state.position,
                        targetPredictions[step],
                        testAccel,
                        1.0f,    // distance weight
                        0.01f,  // control weight
                        0.1f,    // time weight
                        (float)step
                    );
                    totalCost += stepCost;
                    
                    // Early termination if very close
                    float3 diff = state.position - targetPredictions[step];
                    if (length(diff) < 100.0f) {
                        totalCost -= 1000.0f;  // Bonus for interception
                        break;
                    }
                }
                
                if (totalCost < bestControl.cost) {
                    bestControl.cost = totalCost;
                    bestControl.acceleration = testAccel;
                    bestControl.thrust = thrustMag;
                }
            }
        }
    }
    
    return bestControl;
}

// Combined RL + MPC guidance
__host__ __device__ inline float3 rlMpcGuidance(
    float3 interceptorPos,
    float3 interceptorVel,
    float3 targetPos,
    float3 targetVel,
    float3 rlAction,          // High-level RL policy output (direction bias)
    float fuel,
    float thrustMag,
    float rlWeight            // Weight for RL vs MPC (0.0 = pure MPC, 1.0 = pure RL)
) {
    // RL provides high-level direction/intent
    float3 rlDirection = normalize(rlAction);
    
    // Calculate pure pursuit direction
    float3 toTarget = targetPos - interceptorPos;
    float3 pursuitDir = normalize(toTarget);
    
    // Calculate proportional navigation
    float3 pnAccel = physxProportionalNavigation(
        interceptorPos, interceptorVel,
        targetPos, targetVel,
        4.0f  // Navigation gain
    );
    float3 pnDir = length(pnAccel) > 0.01f ? normalize(pnAccel) : pursuitDir;
    
    // Blend RL direction with PN
    float3 combinedDir = normalize(rlDirection * rlWeight + pnDir * (1.0f - rlWeight));
    
    // MPC refines the thrust magnitude
    float optimalThrust = thrustMag;
    if (fuel > 0.0f) {
        float distance = length(toTarget);
        float closingSpeed = -dot(interceptorVel - targetVel, normalize(toTarget));
        
        // Adaptive thrust based on closing geometry
        if (closingSpeed < 0.0f || distance > 5000.0f) {
            optimalThrust = thrustMag;  // Full thrust to catch up
        } else if (distance < 500.0f) {
            optimalThrust = thrustMag * 0.5f;  // Reduce for fine control
        }
    }
    
    return combinedDir * optimalThrust;
}

#endif // PHYSX_ENGINE_CUH
