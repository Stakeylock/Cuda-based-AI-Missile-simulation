#ifndef SIXDOF_DYNAMICS_CUH
#define SIXDOF_DYNAMICS_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"

// ============================================================================
// QUATERNION-BASED 6-DOF DYNAMICS ENGINE
// Based on research document: "High-Fidelity Flight Dynamics"
// Implements singularity-free attitude representation with full ECEF support
// ============================================================================

// ============================================================================
// ATMOSPHERIC MODEL (ISA - for standalone use when physx_engine.cuh not included)
// ============================================================================

#ifndef ATMOSPHERE_FUNCS_DEFINED
#define ATMOSPHERE_FUNCS_DEFINED 1

__host__ __device__ inline float getAirDensity(float altitude) {
    if (altitude < 0.0f) altitude = 0.0f;
    if (altitude > 80000.0f) altitude = 80000.0f;
    
    float temperature = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude;
    if (temperature < 216.65f) temperature = 216.65f;
    
    float pressure = SEA_LEVEL_PRESSURE * 
        powf(temperature / SEA_LEVEL_TEMP, 
             GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * TEMP_LAPSE_RATE));
    
    return (pressure * MOLAR_MASS_AIR) / (GAS_CONSTANT * temperature);
}

__host__ __device__ inline float getSpeedOfSound(float altitude) {
    if (altitude < 0.0f) altitude = 0.0f;
    float temp = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude;
    if (temp < 216.65f) temp = 216.65f;
    return sqrtf(1.4f * 8.314f * temp / 0.029f);
}

#endif // ATMOSPHERE_FUNCS_DEFINED

// ============================================================================
// QUATERNION OPERATIONS
// ============================================================================

// Quaternion structure: q = w + xi + yj + zk = [w, x, y, z]


// Create quaternion from axis-angle representation
__host__ __device__ inline Quaternion quatFromAxisAngle(float3 axis, float angle) {
    float halfAngle = angle * 0.5f;
    float s = sinf(halfAngle);
    float axisLen = length(axis);
    if (axisLen < 1e-6f) {
        return {1.0f, 0.0f, 0.0f, 0.0f};  // Identity
    }
    axis = axis / axisLen;
    return {cosf(halfAngle), axis.x * s, axis.y * s, axis.z * s};
}

// Quaternion multiplication: q1 ⊗ q2
__host__ __device__ inline Quaternion quatMultiply(Quaternion q1, Quaternion q2) {
    Quaternion result;
    result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return result;
}

// Quaternion conjugate
__host__ __device__ inline Quaternion quatConjugate(Quaternion q) {
    return {q.w, -q.x, -q.y, -q.z};
}

// Quaternion normalization
__host__ __device__ inline Quaternion quatNormalize(Quaternion q) {
    float mag = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag < 1e-8f) return {1.0f, 0.0f, 0.0f, 0.0f};
    return {q.w / mag, q.x / mag, q.y / mag, q.z / mag};
}

// Rotate vector by quaternion: v' = q ⊗ v ⊗ q*
__host__ __device__ inline float3 quatRotateVector(Quaternion q, float3 v) {
    // Optimized rotation without full quaternion multiplication
    float3 qv = make_float3(q.x, q.y, q.z);
    float3 uv = cross(qv, v);
    float3 uuv = cross(qv, uv);
    return v + 2.0f * (uv * q.w + uuv);
}

// Convert quaternion to rotation matrix (for Direction Cosine Matrix)
__host__ __device__ inline void quatToRotationMatrix(Quaternion q, float R[9]) {
    float w = q.w, x = q.x, y = q.y, z = q.z;
    
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - w * z);
    R[2] = 2.0f * (x * z + w * y);
    
    R[3] = 2.0f * (x * y + w * z);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - w * x);
    
    R[6] = 2.0f * (x * z - w * y);
    R[7] = 2.0f * (y * z + w * x);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// Quaternion kinematic equation: q_dot = 0.5 * q ⊗ Ω
// Where Ω = [0, ωx, ωy, ωz] is the angular velocity quaternion
__host__ __device__ inline Quaternion quatDerivative(Quaternion q, float3 omega) {
    Quaternion omegaQuat = {0.0f, omega.x, omega.y, omega.z};
    Quaternion result = quatMultiply(q, omegaQuat);
    return {result.w * 0.5f, result.x * 0.5f, result.y * 0.5f, result.z * 0.5f};
}

// ============================================================================
// ECEF COORDINATE SYSTEM
// Earth-Centered Earth-Fixed for long-range ballistic trajectories
// ============================================================================

// WGS-84 Earth parameters
#define EARTH_RADIUS_EQUATORIAL 6378137.0f          // meters
#define EARTH_RADIUS_POLAR 6356752.3142f            // meters
#define EARTH_FLATTENING 0.00335281066474748f       // (a-b)/a
#define EARTH_ECCENTRICITY_SQ 0.00669437999014132f  // e^2
#define EARTH_ROTATION_RATE 7.292115e-5f            // rad/s
#define J2_PERTURBATION 1.08263e-3f                 // J2 oblateness coefficient

// Convert geodetic (lat, lon, alt) to ECEF
__host__ __device__ inline float3 geodeticToECEF(float lat, float lon, float alt) {
    float sinLat = sinf(lat);
    float cosLat = cosf(lat);
    float sinLon = sinf(lon);
    float cosLon = cosf(lon);
    
    float N = EARTH_RADIUS_EQUATORIAL / sqrtf(1.0f - EARTH_ECCENTRICITY_SQ * sinLat * sinLat);
    
    return make_float3(
        (N + alt) * cosLat * cosLon,
        (N + alt) * cosLat * sinLon,
        (N * (1.0f - EARTH_ECCENTRICITY_SQ) + alt) * sinLat
    );
}

// Convert ECEF to geodetic (iterative algorithm)
__host__ __device__ inline void ecefToGeodetic(float3 ecef, float* lat, float* lon, float* alt) {
    float x = ecef.x, y = ecef.y, z = ecef.z;
    
    *lon = atan2f(y, x);
    
    float p = sqrtf(x * x + y * y);
    float theta = atan2f(z * EARTH_RADIUS_EQUATORIAL, p * EARTH_RADIUS_POLAR);
    
    *lat = atan2f(
        z + EARTH_ECCENTRICITY_SQ / (1.0f - EARTH_FLATTENING) * EARTH_RADIUS_POLAR * powf(sinf(theta), 3),
        p - EARTH_ECCENTRICITY_SQ * EARTH_RADIUS_EQUATORIAL * powf(cosf(theta), 3)
    );
    
    float sinLat = sinf(*lat);
    float N = EARTH_RADIUS_EQUATORIAL / sqrtf(1.0f - EARTH_ECCENTRICITY_SQ * sinLat * sinLat);
    *alt = p / cosf(*lat) - N;
}

// Gravity model with J2 perturbation (oblate Earth)
__host__ __device__ inline float3 gravityJ2(float3 r_ecef) {
    float r = length(r_ecef);
    if (r < 1000.0f) r = 1000.0f;  // Prevent singularity
    
    float r2 = r * r;
    float r5 = r2 * r2 * r;
    float re2 = EARTH_RADIUS_EQUATORIAL * EARTH_RADIUS_EQUATORIAL;
    float z2 = r_ecef.z * r_ecef.z;
    
    // Central gravity + J2 perturbation
    float mu_r3 = EARTH_MU / (r2 * r);
    float j2_factor = 1.5f * J2_PERTURBATION * re2 / r2;
    
    float common = mu_r3 * (1.0f + j2_factor * (1.0f - 5.0f * z2 / r2));
    float z_factor = mu_r3 * j2_factor * (3.0f - 5.0f * z2 / r2);
    
    return make_float3(
        -r_ecef.x * common,
        -r_ecef.y * common,
        -r_ecef.z * (common + 2.0f * z_factor)
    );
}

// Simple gravity (flat earth approximation for short range)
__host__ __device__ inline float3 gravitySimple(float altitude) {
    // g varies with altitude: g = g0 * (R / (R + h))^2
    float ratio = EARTH_RADIUS_EQUATORIAL / (EARTH_RADIUS_EQUATORIAL + altitude);
    return make_float3(0.0f, -GRAVITY * ratio * ratio, 0.0f);
}

// ============================================================================
// VARIABLE MASS AND INERTIA (MCI) MODEL
// Critical for accurate missile simulation during boost phase
// ============================================================================





// Initialize mass properties for a typical missile
__host__ __device__ inline MassProperties initMassProperties(
    float totalMass, 
    float propellantFraction,
    float length,
    float radius
) {
    MassProperties mp;
    
    mp.initialMass = totalMass;
    mp.propellantMass = totalMass * propellantFraction;
    mp.mass = totalMass;
    mp.burnRate = mp.propellantMass / 10.0f;  // Assume 10 second burn
    
    // Initial CoM (slightly aft due to motor/propellant)
    mp.centerOfMass = make_float3(0.0f, 0.0f, -length * 0.1f);
    
    // Moments of inertia for cylinder approximation
    float dryMass = totalMass - mp.propellantMass;
    mp.Ixx = dryMass * radius * radius * 0.5f;  // Roll
    mp.Iyy = dryMass * (3.0f * radius * radius + length * length) / 12.0f;  // Pitch
    mp.Izz = mp.Iyy;  // Yaw (symmetric)
    
    mp.Ixy = 0.0f;
    mp.Ixz = 0.0f;
    mp.Iyz = 0.0f;
    
    return mp;
}

// Update mass properties as propellant burns
__host__ __device__ inline void updateMassProperties(MassProperties* mp, float dt) {
    if (mp->propellantMass <= 0.0f) return;
    
    float consumed = fminf(mp->burnRate * dt, mp->propellantMass);
    mp->propellantMass -= consumed;
    mp->mass = mp->initialMass - (mp->initialMass * 0.3f - mp->propellantMass);  // Approx
    
    // CoM shifts forward as propellant depletes
    float burnFraction = 1.0f - mp->propellantMass / (mp->initialMass * 0.3f);
    mp->centerOfMass.z = -0.3f * (1.0f - burnFraction);  // Shifts toward nose
    
    // Inertia decreases as mass decreases
    float massRatio = mp->mass / mp->initialMass;
    mp->Iyy *= massRatio;
    mp->Izz *= massRatio;
}

// ============================================================================
// FULL 6-DOF STATE VECTOR
// x = [r_ECEF, v_ECEF, q_BI, ω_B, m] ∈ R^14
// ============================================================================



// ============================================================================
// AERODYNAMIC DATABASE (Multi-dimensional lookup)
// C_A, C_N, C_m = f(Mach, Alpha, Beta, Delta)
// ============================================================================

// Structure for aerodynamic coefficients at a single flight condition
struct AeroCoefficients {
    float CA;    // Axial force coefficient (drag direction)
    float CY;    // Side force coefficient
    float CN;    // Normal force coefficient (lift direction)
    float Cl;    // Rolling moment coefficient
    float Cm;    // Pitching moment coefficient
    float Cn;    // Yawing moment coefficient
    float CNA;   // Normal force slope (dCN/dalpha)
    float CMA;   // Pitching moment slope (stability derivative)
};

// Lookup aerodynamic coefficients from database (texture-backed)
// In production, this would sample from CUDA texture memory
__host__ __device__ inline AeroCoefficients lookupAeroCoefficients(
    float mach, 
    float alpha,  // angle of attack (rad)
    float beta,   // sideslip (rad)
    float delta   // control deflection (rad)
) {
    AeroCoefficients coef;
    
    // Convert to degrees for easier coefficient definition
    float alphaDeg = alpha * 180.0f / M_PI;
    float betaDeg = beta * 180.0f / M_PI;
    float deltaDeg = delta * 180.0f / M_PI;
    
    // Base coefficients (generic missile shape)
    float CD0 = 0.2f;   // Zero-lift drag
    float CLalpha = 0.05f;  // Lift curve slope (per degree)
    
    // Mach number effects (wave drag)
    float machFactor = 1.0f;
    if (mach < 0.8f) {
        machFactor = 1.0f;
    } else if (mach < 1.2f) {
        // Transonic drag rise
        float t = (mach - 0.8f) / 0.4f;
        machFactor = 1.0f + 0.8f * sinf(t * M_PI);
    } else if (mach < 3.0f) {
        // Supersonic
        machFactor = 1.0f + 0.4f / (mach - 0.5f);
    } else {
        // Hypersonic
        machFactor = 0.6f;
        CLalpha *= 0.5f;  // Reduced lift effectiveness
    }
    
    // Axial force (drag)
    coef.CA = CD0 * machFactor + 0.01f * alphaDeg * alphaDeg;
    
    // Normal force (lift)
    coef.CN = CLalpha * alphaDeg * (1.0f - 0.01f * fabsf(alphaDeg));  // Stall effect
    coef.CNA = CLalpha;
    
    // Side force
    coef.CY = 0.03f * betaDeg;
    
    // Pitching moment (nose-down is positive)
    float xcp = 0.3f;  // Center of pressure location (fraction from nose)
    float xcg = 0.4f;  // Center of gravity location
    float staticMargin = xcg - xcp;
    coef.Cm = -coef.CN * staticMargin + 0.02f * deltaDeg;  // Control effectiveness
    coef.CMA = -coef.CNA * staticMargin;
    
    // Roll moment (from asymmetric fins or aileron)
    coef.Cl = 0.01f * deltaDeg;
    
    // Yaw moment
    coef.Cn = -coef.CY * staticMargin;
    
    return coef;
}

// ============================================================================
// HYPERSONIC GLIDE VEHICLE (HGV) AERODYNAMICS
// Waverider-specific aerodynamic model
// ============================================================================

struct WaveriderAero {
    float LD_max;           // Maximum lift-to-drag ratio
    float LD_mach_factor;   // L/D degradation with Mach
    float bankAngle;        // Current bank angle for lateral maneuver
    float equilibriumGlideGamma;  // Flight path angle for equilibrium glide
};

__host__ __device__ inline AeroCoefficients lookupWaveriderAero(
    float mach,
    float alpha,
    float bankAngle
) {
    AeroCoefficients coef;
    
    // Waverider at hypersonic speeds (Mach 5-20)
    // Based on HTV-2 and similar vehicle data
    
    // L/D degrades at very high Mach numbers
    float LD_ratio = 2.5f;  // Typical hypersonic L/D
    if (mach > 10.0f) {
        LD_ratio = 2.5f - 0.1f * (mach - 10.0f);  // Degrades above Mach 10
    }
    LD_ratio = fmaxf(LD_ratio, 1.5f);
    
    // At optimal alpha (typically 5-10 degrees for waverider)
    float optimalAlpha = 7.0f * M_PI / 180.0f;
    float alphaFactor = 1.0f - 0.5f * fabsf(alpha - optimalAlpha) / optimalAlpha;
    alphaFactor = fmaxf(alphaFactor, 0.3f);
    
    // Lift coefficient (hypersonic Newtonian theory approximation)
    coef.CN = 0.02f * sinf(2.0f * alpha) * alphaFactor;
    
    // Drag coefficient
    coef.CA = coef.CN / LD_ratio + 0.005f;  // Skin friction + pressure drag
    
    // Bank angle affects lateral force distribution
    coef.CY = coef.CN * sinf(bankAngle);
    coef.CN = coef.CN * cosf(bankAngle);
    
    // Moment coefficients (waveriders are marginally stable)
    coef.Cm = -0.001f * (alpha - optimalAlpha);  // Weak pitch stability
    coef.CMA = -0.01f;
    coef.Cl = 0.0f;
    coef.Cn = -0.001f * bankAngle;
    coef.CNA = 0.02f;
    
    return coef;
}

// ============================================================================
// 6-DOF EQUATIONS OF MOTION
// Newton-Euler formulation with full aerodynamic forces and moments
// ============================================================================

// Compute all forces and moments on the vehicle
__host__ __device__ inline void compute6DOFDerivatives(
    State6DOF* state,
    float3 thrustBody,       // Thrust vector in body frame
    float3 controlMoment,    // Control moments from fins
    float Sref,              // Reference area
    float Lref,              // Reference length
    float dt,
    State6DOF* derivative
) {
    // Get atmospheric conditions
    float altitude = state->altitude;
    float airDensity = getAirDensity(altitude);
    float speedOfSound = getSpeedOfSound(altitude);
    
    // Velocity in body frame
    float3 vel_inertial = state->velocity;
    Quaternion q_inv = quatConjugate(state->attitude);
    float3 vel_body = quatRotateVector(q_inv, vel_inertial);
    
    float V = length(vel_body);
    if (V < 1.0f) V = 1.0f;
    
    // Compute aerodynamic angles
    float alpha = atan2f(vel_body.y, vel_body.x);  // Angle of attack
    float beta = asinf(fmaxf(fminf(vel_body.z / V, 1.0f), -1.0f));  // Sideslip
    
    state->angleOfAttack = alpha;
    state->sideslipAngle = beta;
    state->machNumber = V / speedOfSound;
    state->velocity_body = vel_body;
    
    // Dynamic pressure
    float Q = 0.5f * airDensity * V * V;
    state->dynamicPressure = Q;
    
    // Lookup aerodynamic coefficients
    AeroCoefficients aero = lookupAeroCoefficients(
        state->machNumber, alpha, beta, 0.0f
    );
    
    // Aerodynamic forces in stability frame
    float3 F_aero_stab = make_float3(
        -aero.CA * Q * Sref,   // Axial (drag)
        aero.CY * Q * Sref,    // Side
        -aero.CN * Q * Sref    // Normal (lift)
    );
    
    // Transform stability forces to body frame
    // (Simple approximation: stability ≈ body for small alpha)
    float3 F_aero_body = F_aero_stab;
    
    // Aerodynamic moments in body frame
    float3 M_aero = make_float3(
        aero.Cl * Q * Sref * Lref,   // Roll
        aero.Cm * Q * Sref * Lref,   // Pitch
        aero.Cn * Q * Sref * Lref    // Yaw
    );
    
    // Total body forces
    float3 F_body = F_aero_body + thrustBody;
    
    // Transform to inertial frame
    float3 F_inertial = quatRotateVector(state->attitude, F_body);
    
    // Add gravity (in inertial/ECEF frame)
    float3 gravity = gravitySimple(altitude);  // Use J2 for long range
    
    // Coriolis effect (for ECEF)
    // v_dot = F/m - 2*Omega x v - Omega x (Omega x r) + g
    // For simplicity in local simulation, we skip Coriolis (NED assumption)
    
    // Translational derivatives
    derivative->velocity = F_inertial / state->massProps.mass + gravity;
    derivative->position = state->velocity;
    
    // Total moments
    float3 M_total = M_aero + controlMoment;
    
    // Rotational dynamics: I * omega_dot + omega x (I * omega) = M
    // For principal axes (Ixy = Ixz = Iyz = 0):
    float Ixx = state->massProps.Ixx;
    float Iyy = state->massProps.Iyy;
    float Izz = state->massProps.Izz;
    float3 omega = state->angularVelocity;
    
    float3 omega_cross_Iomega = make_float3(
        omega.y * omega.z * (Izz - Iyy),
        omega.z * omega.x * (Ixx - Izz),
        omega.x * omega.y * (Iyy - Ixx)
    );
    
    derivative->angularVelocity = make_float3(
        (M_total.x - omega_cross_Iomega.x) / Ixx,
        (M_total.y - omega_cross_Iomega.y) / Iyy,
        (M_total.z - omega_cross_Iomega.z) / Izz
    );
    
    // Quaternion derivative
    Quaternion q_dot = quatDerivative(state->attitude, omega);
    derivative->attitude = q_dot;
    
    // Mass derivative (propellant consumption)
    derivative->massProps.mass = -state->massProps.burnRate;
}

// ============================================================================
// RK4 INTEGRATOR FOR 6-DOF STATE
// Higher-order integration for numerical accuracy
// ============================================================================

__host__ __device__ inline void integrate6DOF_RK4(
    State6DOF* state,
    float3 thrustBody,
    float3 controlMoment,
    float Sref,
    float Lref,
    float dt
) {
    State6DOF k1, k2, k3, k4;
    State6DOF temp;
    
    // k1 = f(t, y)
    compute6DOFDerivatives(state, thrustBody, controlMoment, Sref, Lref, dt, &k1);
    
    // k2 = f(t + dt/2, y + dt/2 * k1)
    temp = *state;
    temp.position = state->position + k1.position * (dt * 0.5f);
    temp.velocity = state->velocity + k1.velocity * (dt * 0.5f);
    temp.attitude.w = state->attitude.w + k1.attitude.w * (dt * 0.5f);
    temp.attitude.x = state->attitude.x + k1.attitude.x * (dt * 0.5f);
    temp.attitude.y = state->attitude.y + k1.attitude.y * (dt * 0.5f);
    temp.attitude.z = state->attitude.z + k1.attitude.z * (dt * 0.5f);
    temp.attitude = quatNormalize(temp.attitude);
    temp.angularVelocity = state->angularVelocity + k1.angularVelocity * (dt * 0.5f);
    temp.altitude = temp.position.y;
    compute6DOFDerivatives(&temp, thrustBody, controlMoment, Sref, Lref, dt, &k2);
    
    // k3 = f(t + dt/2, y + dt/2 * k2)
    temp = *state;
    temp.position = state->position + k2.position * (dt * 0.5f);
    temp.velocity = state->velocity + k2.velocity * (dt * 0.5f);
    temp.attitude.w = state->attitude.w + k2.attitude.w * (dt * 0.5f);
    temp.attitude.x = state->attitude.x + k2.attitude.x * (dt * 0.5f);
    temp.attitude.y = state->attitude.y + k2.attitude.y * (dt * 0.5f);
    temp.attitude.z = state->attitude.z + k2.attitude.z * (dt * 0.5f);
    temp.attitude = quatNormalize(temp.attitude);
    temp.angularVelocity = state->angularVelocity + k2.angularVelocity * (dt * 0.5f);
    temp.altitude = temp.position.y;
    compute6DOFDerivatives(&temp, thrustBody, controlMoment, Sref, Lref, dt, &k3);
    
    // k4 = f(t + dt, y + dt * k3)
    temp = *state;
    temp.position = state->position + k3.position * dt;
    temp.velocity = state->velocity + k3.velocity * dt;
    temp.attitude.w = state->attitude.w + k3.attitude.w * dt;
    temp.attitude.x = state->attitude.x + k3.attitude.x * dt;
    temp.attitude.y = state->attitude.y + k3.attitude.y * dt;
    temp.attitude.z = state->attitude.z + k3.attitude.z * dt;
    temp.attitude = quatNormalize(temp.attitude);
    temp.angularVelocity = state->angularVelocity + k3.angularVelocity * dt;
    temp.altitude = temp.position.y;
    compute6DOFDerivatives(&temp, thrustBody, controlMoment, Sref, Lref, dt, &k4);
    
    // y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    float dt6 = dt / 6.0f;
    state->position = state->position + (k1.position + k2.position * 2.0f + k3.position * 2.0f + k4.position) * dt6;
    state->velocity = state->velocity + (k1.velocity + k2.velocity * 2.0f + k3.velocity * 2.0f + k4.velocity) * dt6;
    
    state->attitude.w += (k1.attitude.w + 2.0f * k2.attitude.w + 2.0f * k3.attitude.w + k4.attitude.w) * dt6;
    state->attitude.x += (k1.attitude.x + 2.0f * k2.attitude.x + 2.0f * k3.attitude.x + k4.attitude.x) * dt6;
    state->attitude.y += (k1.attitude.y + 2.0f * k2.attitude.y + 2.0f * k3.attitude.y + k4.attitude.y) * dt6;
    state->attitude.z += (k1.attitude.z + 2.0f * k2.attitude.z + 2.0f * k3.attitude.z + k4.attitude.z) * dt6;
    state->attitude = quatNormalize(state->attitude);
    
    state->angularVelocity = state->angularVelocity + (k1.angularVelocity + k2.angularVelocity * 2.0f + k3.angularVelocity * 2.0f + k4.angularVelocity) * dt6;
    
    // Update mass properties
    updateMassProperties(&state->massProps, dt);
    
    // Update derived quantities
    state->altitude = state->position.y;
    state->acceleration = k1.velocity;  // Store for logging
}

#endif // SIXDOF_DYNAMICS_CUH
