#ifndef SENSOR_SIMULATION_CUH
#define SENSOR_SIMULATION_CUH

#include "../core/config.h"
#include "cuda_utils.cuh"
#include "sixdof_dynamics.cuh"

// ============================================================================
// GPU-ACCELERATED RADAR CROSS SECTION (RCS) AND MICRO-DOPPLER SIMULATION
// Based on research document: "Sensor Simulation: The Eyes of the System"
// Enables training of AI discrimination algorithms (warhead vs decoy)
// ============================================================================

// ============================================================================
// SCATTERER MODEL FOR RCS CALCULATION
// Point-scatterer model for real-time GPU computation
// ============================================================================

#define MAX_SCATTERERS 64  // Maximum scattering centers per target

// Individual scattering point on target
struct Scatterer {
    float3 position;       // Position relative to target CoM (body frame)
    float amplitude;       // Scattering amplitude (sqrt of RCS contribution)
    float phase;           // Base phase shift
    float specularFactor;  // How specular vs diffuse (0-1)
};

// Complete target electromagnetic signature
struct TargetSignature {
    Scatterer scatterers[MAX_SCATTERERS];
    int numScatterers;
    
    // Physical properties affecting signature
    float length;          // Target length (m)
    float radius;          // Target radius (m)
    float coneAngle;       // Nose cone half-angle (rad) for reentry vehicles
    
    // Material properties
    float conductivity;    // Surface conductivity
    float roughness;       // Surface roughness for diffuse scatter
    
    // Dynamic properties (for micro-Doppler)
    float3 spinAxis;       // Axis of rotation (body frame)
    float spinRate;        // Angular velocity (rad/s)
    float wobbleAngle;     // Precession/nutation angle (rad)
    float wobbleRate;      // Precession rate (rad/s)
};

// ============================================================================
// TARGET TYPE DEFINITIONS (for discrimination training)
// ============================================================================

enum TargetType {
    TARGET_WARHEAD = 0,     // High-mass, heavy precession, cone RCS
    TARGET_DECOY_BALLOON,   // Light, tumbles chaotically, sphere RCS
    TARGET_DECOY_CONE,      // Light cone decoy, different wobble signature
    TARGET_DEBRIS,          // Irregular tumbling, small RCS
    TARGET_HGV,             // Hypersonic glider, waverider RCS
    TARGET_BOOSTER          // Rocket body, large RCS, specific motion
};

// Initialize signature for different target types
__host__ __device__ inline TargetSignature initTargetSignature(TargetType type) {
    TargetSignature sig;
    memset(&sig, 0, sizeof(TargetSignature));
    
    switch (type) {
        case TARGET_WARHEAD: {
            // Conical reentry vehicle with spherical nose
            sig.length = 2.0f;
            sig.radius = 0.3f;
            sig.coneAngle = 15.0f * M_PI / 180.0f;
            sig.conductivity = 1e6f;  // Metallic
            sig.roughness = 0.01f;
            
            // Scatterers: nose tip, body, base
            sig.numScatterers = 8;
            
            // Nose tip (strong specular return)
            sig.scatterers[0] = {{0.0f, 0.0f, sig.length / 2.0f}, 0.5f, 0.0f, 0.9f};
            
            // Body creeping wave returns
            for (int i = 1; i < 5; i++) {
                float z = sig.length * (0.25f - 0.15f * i);
                float r = sig.radius * (0.8f + 0.1f * i);
                sig.scatterers[i] = {{r, 0.0f, z}, 0.2f, (float)i * 0.5f, 0.5f};
            }
            
            // Base edge diffraction
            sig.scatterers[5] = {{0.0f, 0.0f, -sig.length / 2.0f}, 0.4f, M_PI, 0.3f};
            sig.scatterers[6] = {{sig.radius, 0.0f, -sig.length / 2.0f}, 0.15f, M_PI, 0.2f};
            sig.scatterers[7] = {{-sig.radius, 0.0f, -sig.length / 2.0f}, 0.15f, M_PI, 0.2f};
            
            // Heavy wobble/precession (mass asymmetry)
            sig.spinAxis = make_float3(0.0f, 0.0f, 1.0f);
            sig.spinRate = 2.0f;        // 2 rad/s spin
            sig.wobbleAngle = 10.0f * M_PI / 180.0f;  // 10 degree precession
            sig.wobbleRate = 0.5f;      // Slow precession
            break;
        }
        
        case TARGET_DECOY_BALLOON: {
            // Spherical mylar balloon
            sig.length = 1.0f;
            sig.radius = 0.5f;
            sig.coneAngle = 0.0f;
            sig.conductivity = 1e4f;   // Metalized plastic
            sig.roughness = 0.1f;
            
            // Spherical scatterer pattern
            sig.numScatterers = 12;
            
            // Main specular point
            sig.scatterers[0] = {{0.0f, 0.0f, sig.radius}, 0.8f, 0.0f, 0.95f};
            
            // Distributed points around sphere
            for (int i = 1; i < 12; i++) {
                float theta = i * M_PI / 6.0f;
                float phi = i * M_PI / 3.0f;
                float3 pos = make_float3(
                    sig.radius * sinf(theta) * cosf(phi),
                    sig.radius * sinf(theta) * sinf(phi),
                    sig.radius * cosf(theta)
                );
                sig.scatterers[i] = {pos, 0.1f, (float)i * 0.3f, 0.7f};
            }
            
            // Light tumbling (chaotic, high rate)
            sig.spinAxis = make_float3(0.3f, 0.5f, 0.8f);
            sig.spinRate = 5.0f;        // Fast tumbling
            sig.wobbleAngle = 45.0f * M_PI / 180.0f;  // Large wobble
            sig.wobbleRate = 3.0f;      // Fast precession
            break;
        }
        
        case TARGET_DECOY_CONE: {
            // Lightweight cone decoy
            sig.length = 1.5f;
            sig.radius = 0.2f;
            sig.coneAngle = 12.0f * M_PI / 180.0f;
            sig.conductivity = 5e5f;
            sig.roughness = 0.02f;
            
            sig.numScatterers = 5;
            sig.scatterers[0] = {{0.0f, 0.0f, sig.length / 2.0f}, 0.3f, 0.0f, 0.85f};
            sig.scatterers[1] = {{sig.radius * 0.5f, 0.0f, 0.0f}, 0.15f, 0.5f, 0.4f};
            sig.scatterers[2] = {{-sig.radius * 0.5f, 0.0f, 0.0f}, 0.15f, 0.5f, 0.4f};
            sig.scatterers[3] = {{0.0f, sig.radius * 0.5f, 0.0f}, 0.15f, 0.7f, 0.4f};
            sig.scatterers[4] = {{0.0f, 0.0f, -sig.length / 2.0f}, 0.2f, M_PI, 0.3f};
            
            // Different wobble than warhead (lighter = faster, different frequency)
            sig.spinAxis = make_float3(0.0f, 0.0f, 1.0f);
            sig.spinRate = 4.0f;        // Faster spin
            sig.wobbleAngle = 20.0f * M_PI / 180.0f;
            sig.wobbleRate = 2.0f;
            break;
        }
        
        case TARGET_HGV: {
            // Waverider/hypersonic glide vehicle
            sig.length = 5.0f;
            sig.radius = 1.0f;
            sig.coneAngle = 8.0f * M_PI / 180.0f;
            sig.conductivity = 2e6f;
            sig.roughness = 0.005f;
            
            // Complex waverider shape
            sig.numScatterers = 16;
            
            // Leading edges (primary returns)
            sig.scatterers[0] = {{-sig.radius, 0.0f, sig.length * 0.3f}, 0.4f, 0.0f, 0.7f};
            sig.scatterers[1] = {{sig.radius, 0.0f, sig.length * 0.3f}, 0.4f, 0.0f, 0.7f};
            
            // Body facets
            for (int i = 2; i < 10; i++) {
                float z = sig.length * (0.2f - 0.05f * i);
                float x = sig.radius * (0.8f + 0.1f * (i % 2)) * (i % 2 == 0 ? 1.0f : -1.0f);
                sig.scatterers[i] = {{x, 0.0f, z}, 0.2f, (float)i * 0.4f, 0.5f};
            }
            
            // Control surfaces
            for (int i = 10; i < 16; i++) {
                float3 pos = make_float3(
                    sig.radius * 0.5f * ((i - 10) % 2 == 0 ? 1.0f : -1.0f),
                    0.1f * (i - 10 - 2),
                    -sig.length * 0.3f
                );
                sig.scatterers[i] = {pos, 0.1f, (float)i * 0.2f, 0.3f};
            }
            
            // HGV has controlled flight - minimal wobble
            sig.spinAxis = make_float3(1.0f, 0.0f, 0.0f);
            sig.spinRate = 0.1f;        // Very slow roll
            sig.wobbleAngle = 2.0f * M_PI / 180.0f;  // Minimal
            sig.wobbleRate = 0.0f;
            break;
        }
        
        default: {
            // Generic debris
            sig.length = 1.0f;
            sig.radius = 0.3f;
            sig.numScatterers = 4;
            sig.scatterers[0] = {{0.0f, 0.0f, 0.5f}, 0.3f, 0.0f, 0.4f};
            sig.scatterers[1] = {{0.2f, 0.1f, -0.2f}, 0.2f, 0.5f, 0.3f};
            sig.scatterers[2] = {{-0.15f, 0.2f, 0.1f}, 0.15f, 1.0f, 0.3f};
            sig.scatterers[3] = {{0.0f, -0.2f, -0.3f}, 0.1f, 1.5f, 0.2f};
            
            sig.spinAxis = make_float3(0.5f, 0.5f, 0.707f);
            sig.spinRate = 3.0f;
            sig.wobbleAngle = 30.0f * M_PI / 180.0f;
            sig.wobbleRate = 1.5f;
            break;
        }
    }
    
    return sig;
}

// ============================================================================
// RCS CALCULATION (Coherent summation of scatterer returns)
// ============================================================================

// Radar parameters
#define RADAR_FREQUENCY 10e9f        // X-band (10 GHz)
#define RADAR_WAVELENGTH 0.03f       // 3 cm wavelength
#define SPEED_OF_LIGHT 3e8f

// Complex number for phase calculations
struct Complex {
    float real;
    float imag;
};

__host__ __device__ inline Complex complexAdd(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

__host__ __device__ inline Complex complexMultiply(Complex a, Complex b) {
    return {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

__host__ __device__ inline float complexMagnitude(Complex c) {
    return sqrtf(c.real * c.real + c.imag * c.imag);
}

__host__ __device__ inline Complex complexFromPolar(float mag, float phase) {
    return {mag * cosf(phase), mag * sinf(phase)};
}

// Calculate RCS and return signal for a target
// Returns: RCS in square meters, fills IQ signal components
__host__ __device__ inline float calculateRCS(
    TargetSignature* sig,
    Quaternion targetAttitude,
    float3 radarLOS,           // Line of sight unit vector (from radar to target)
    float time,                // For micro-Doppler
    Complex* iqSignal          // Output I/Q signal (for micro-Doppler processing)
) {
    Complex totalReturn = {0.0f, 0.0f};
    
    // Compute current attitude including spin and wobble
    float spinPhase = sig->spinRate * time;
    float wobblePhase = sig->wobbleRate * time;
    
    // Apply spin around body axis
    Quaternion spinQuat = quatFromAxisAngle(sig->spinAxis, spinPhase);
    Quaternion currentAtt = quatMultiply(targetAttitude, spinQuat);
    
    // Apply wobble/precession
    float3 wobbleAxis = make_float3(cosf(wobblePhase), sinf(wobblePhase), 0.0f);
    Quaternion wobbleQuat = quatFromAxisAngle(wobbleAxis, sig->wobbleAngle * sinf(wobblePhase));
    currentAtt = quatMultiply(wobbleQuat, currentAtt);
    currentAtt = quatNormalize(currentAtt);
    
    // Transform radar LOS to body frame
    Quaternion attInv = quatConjugate(currentAtt);
    float3 losBody = quatRotateVector(attInv, radarLOS);
    
    // Sum contributions from all scatterers
    for (int i = 0; i < sig->numScatterers; i++) {
        Scatterer* scat = &sig->scatterers[i];
        
        // Transform scatterer position to inertial frame
        float3 scatPosInertial = quatRotateVector(currentAtt, scat->position);
        
        // Phase shift from position along LOS
        float pathLength = dot(scatPosInertial, radarLOS);
        float phase = scat->phase + 4.0f * M_PI * pathLength / RADAR_WAVELENGTH;
        
        // Amplitude based on aspect angle and specular factor
        float cosAspect = fabsf(dot(normalize(scat->position), losBody));
        float aspectFactor = scat->specularFactor * powf(cosAspect, 4.0f) + 
                            (1.0f - scat->specularFactor) * cosAspect;
        
        float amplitude = scat->amplitude * aspectFactor;
        
        // Add to coherent sum
        Complex scatReturn = complexFromPolar(amplitude, phase);
        totalReturn = complexAdd(totalReturn, scatReturn);
    }
    
    // Output I/Q signal
    if (iqSignal) {
        *iqSignal = totalReturn;
    }
    
    // RCS = |E_scattered|^2 / |E_incident|^2 * 4*pi*R^2
    // For normalized incident field, RCS proportional to |return|^2
    float rcs = complexMagnitude(totalReturn);
    rcs = rcs * rcs;  // Square for power
    
    return rcs;
}

// ============================================================================
// MICRO-DOPPLER SIGNATURE CALCULATION
// Key discriminator between warheads and decoys
// ============================================================================

#define MICRO_DOPPLER_SAMPLES 256  // Number of time samples for spectrogram

struct MicroDopplerSignature {
    float velocities[MAX_SCATTERERS];      // Radial velocity of each scatterer
    float amplitudes[MAX_SCATTERERS];      // Signal amplitude from each
    float totalBandwidth;                  // Total micro-Doppler bandwidth
    float mainBodyVelocity;               // Bulk velocity component
    float wobbleSignature;                // Signature strength from wobble
    
    // Time-frequency features for ML
    float centroid;                        // Spectral centroid
    float bandwidth;                       // Spectral bandwidth  
    float periodicity;                     // Periodicity measure (wobble frequency)
};

// Calculate micro-Doppler velocities for each scatterer
__host__ __device__ inline MicroDopplerSignature calculateMicroDoppler(
    TargetSignature* sig,
    Quaternion targetAttitude,
    float3 targetVelocity,     // Bulk velocity
    float3 targetAngularVel,   // Angular velocity (body frame)
    float3 radarLOS,           // LOS from radar to target
    float time
) {
    MicroDopplerSignature md;
    memset(&md, 0, sizeof(MicroDopplerSignature));
    
    // Bulk velocity component (main Doppler)
    md.mainBodyVelocity = -dot(targetVelocity, radarLOS);
    
    // Compute attitude with spin and wobble
    float spinPhase = sig->spinRate * time;
    float wobblePhase = sig->wobbleRate * time;
    
    Quaternion spinQuat = quatFromAxisAngle(sig->spinAxis, spinPhase);
    Quaternion currentAtt = quatMultiply(targetAttitude, spinQuat);
    
    float3 wobbleAxis = make_float3(cosf(wobblePhase), sinf(wobblePhase), 0.0f);
    Quaternion wobbleQuat = quatFromAxisAngle(wobbleAxis, sig->wobbleAngle * sinf(wobblePhase));
    currentAtt = quatMultiply(wobbleQuat, currentAtt);
    currentAtt = quatNormalize(currentAtt);
    
    // Total angular velocity including wobble
    float3 totalOmega = targetAngularVel;
    totalOmega = totalOmega + sig->spinAxis * sig->spinRate;
    
    // Add wobble contribution
    float3 wobbleOmegaBody = wobbleAxis * sig->wobbleRate * sinf(sig->wobbleAngle);
    totalOmega = totalOmega + wobbleOmegaBody;
    
    float maxVel = 0.0f;
    float minVel = 0.0f;
    float weightedSum = 0.0f;
    float totalWeight = 0.0f;
    
    // Calculate radial velocity of each scatterer
    for (int i = 0; i < sig->numScatterers; i++) {
        Scatterer* scat = &sig->scatterers[i];
        
        // Transform scatterer position to inertial
        float3 scatPosInertial = quatRotateVector(currentAtt, scat->position);
        
        // Velocity from rotation: v = ω × r
        float3 rotationalVel = cross(quatRotateVector(currentAtt, totalOmega), scatPosInertial);
        
        // Total velocity of scatterer
        float3 scatVelTotal = targetVelocity + rotationalVel;
        
        // Radial component (positive = approaching radar)
        float radialVel = -dot(scatVelTotal, radarLOS);
        
        md.velocities[i] = radialVel;
        md.amplitudes[i] = scat->amplitude;
        
        // Track bandwidth
        if (radialVel > maxVel) maxVel = radialVel;
        if (radialVel < minVel) minVel = radialVel;
        
        // Weighted centroid calculation
        weightedSum += radialVel * scat->amplitude;
        totalWeight += scat->amplitude;
    }
    
    // Micro-Doppler bandwidth
    md.totalBandwidth = maxVel - minVel;
    
    // Spectral features
    md.centroid = (totalWeight > 0.0f) ? weightedSum / totalWeight : md.mainBodyVelocity;
    md.bandwidth = md.totalBandwidth;
    
    // Periodicity from wobble rate
    md.periodicity = sig->wobbleRate / (2.0f * M_PI);  // Hz
    
    // Wobble signature strength (key discriminator)
    // Warheads have consistent, periodic wobble; decoys are chaotic
    md.wobbleSignature = sig->wobbleAngle * sig->wobbleRate * sig->length;
    
    return md;
}

// ============================================================================
// FEATURE EXTRACTION FOR ML-BASED DISCRIMINATION
// ============================================================================

#define NUM_DISCRIMINATION_FEATURES 16

struct DiscriminationFeatures {
    // RCS features
    float meanRCS;
    float rcsVariance;
    float rcsMin;
    float rcsMax;
    
    // Micro-Doppler features
    float microDopplerBandwidth;
    float spectralCentroid;
    float spectralVariance;
    float periodicity;
    float periodicityStrength;
    
    // Motion features
    float wobbleRate;
    float spinRate;
    float motionRegularity;   // How periodic/predictable the motion is
    
    // Physical features (if radar can estimate)
    float estimatedMass;       // From ballistic coefficient
    float estimatedLD;         // From glide performance
    float thermalSignature;    // IR if available
    
    // Confidence
    float classificationConfidence;
};

// Extract features from a sequence of radar observations
__host__ __device__ inline DiscriminationFeatures extractDiscriminationFeatures(
    float* rcsHistory,         // Array of RCS samples over time
    MicroDopplerSignature* mdHistory,  // Micro-Doppler history
    int numSamples,
    float sampleInterval
) {
    DiscriminationFeatures features;
    memset(&features, 0, sizeof(DiscriminationFeatures));
    
    if (numSamples < 2) return features;
    
    // RCS statistics
    float sumRCS = 0.0f;
    float sumRCS2 = 0.0f;
    features.rcsMin = rcsHistory[0];
    features.rcsMax = rcsHistory[0];
    
    for (int i = 0; i < numSamples; i++) {
        sumRCS += rcsHistory[i];
        sumRCS2 += rcsHistory[i] * rcsHistory[i];
        if (rcsHistory[i] < features.rcsMin) features.rcsMin = rcsHistory[i];
        if (rcsHistory[i] > features.rcsMax) features.rcsMax = rcsHistory[i];
    }
    
    features.meanRCS = sumRCS / numSamples;
    features.rcsVariance = sumRCS2 / numSamples - features.meanRCS * features.meanRCS;
    
    // Micro-Doppler statistics
    float sumBW = 0.0f, sumCent = 0.0f, sumPer = 0.0f;
    float sumCent2 = 0.0f;
    
    for (int i = 0; i < numSamples; i++) {
        sumBW += mdHistory[i].totalBandwidth;
        sumCent += mdHistory[i].centroid;
        sumCent2 += mdHistory[i].centroid * mdHistory[i].centroid;
        sumPer += mdHistory[i].periodicity;
    }
    
    features.microDopplerBandwidth = sumBW / numSamples;
    features.spectralCentroid = sumCent / numSamples;
    features.spectralVariance = sumCent2 / numSamples - features.spectralCentroid * features.spectralCentroid;
    features.periodicity = sumPer / numSamples;
    
    // Motion regularity (lower variance = more regular = more likely warhead)
    // Warheads have predictable precession; decoys tumble chaotically
    float centroidVar = features.spectralVariance;
    float bwVar = 0.0f;
    for (int i = 0; i < numSamples; i++) {
        float diff = mdHistory[i].totalBandwidth - features.microDopplerBandwidth;
        bwVar += diff * diff;
    }
    bwVar /= numSamples;
    
    features.motionRegularity = 1.0f / (1.0f + sqrtf(centroidVar + bwVar));
    
    // Periodicity strength (detect consistent wobble frequency)
    // This would use FFT in production; simplified here
    float periodicityScore = 0.0f;
    float expectedPeriod = 1.0f / (features.periodicity + 0.01f);
    int periodSamples = (int)(expectedPeriod / sampleInterval);
    if (periodSamples > 1 && periodSamples < numSamples / 2) {
        for (int i = 0; i < numSamples - periodSamples; i++) {
            float corr = mdHistory[i].centroid * mdHistory[i + periodSamples].centroid;
            periodicityScore += corr;
        }
        periodicityScore /= (numSamples - periodSamples);
    }
    features.periodicityStrength = fabsf(periodicityScore);
    
    // Extract wobble and spin rates from micro-Doppler
    features.wobbleRate = mdHistory[numSamples / 2].periodicity * 2.0f * M_PI;
    features.spinRate = features.microDopplerBandwidth / (mdHistory[0].totalBandwidth + 0.01f);
    
    return features;
}

// ============================================================================
// NEURAL NETWORK CLASSIFIER FOR TARGET DISCRIMINATION
// ============================================================================

// Simple MLP for target classification (would use CNN on spectrograms in production)
struct DiscriminatorNetwork {
    float weights1[NUM_DISCRIMINATION_FEATURES * 64];
    float bias1[64];
    float weights2[64 * 32];
    float bias2[32];
    float weights3[32 * 6];  // 6 target types
    float bias3[6];
};

// Forward pass for discrimination
__host__ __device__ inline void classifyTarget(
    DiscriminatorNetwork* net,
    DiscriminationFeatures* features,
    float* probabilities  // Output: probability for each target type
) {
    // Normalize features to input vector
    float input[NUM_DISCRIMINATION_FEATURES] = {
        features->meanRCS / 10.0f,
        sqrtf(features->rcsVariance) / 5.0f,
        features->rcsMin / 5.0f,
        features->rcsMax / 20.0f,
        features->microDopplerBandwidth / 100.0f,
        features->spectralCentroid / 500.0f,
        sqrtf(features->spectralVariance) / 50.0f,
        features->periodicity,
        features->periodicityStrength,
        features->wobbleRate / 10.0f,
        features->spinRate / 10.0f,
        features->motionRegularity,
        0.0f, 0.0f, 0.0f, 0.0f  // Placeholder for additional features
    };
    
    // Layer 1
    float hidden1[64];
    for (int i = 0; i < 64; i++) {
        float sum = net->bias1[i];
        for (int j = 0; j < NUM_DISCRIMINATION_FEATURES; j++) {
            sum += input[j] * net->weights1[j * 64 + i];
        }
        hidden1[i] = fmaxf(0.0f, sum);  // ReLU
    }
    
    // Layer 2
    float hidden2[32];
    for (int i = 0; i < 32; i++) {
        float sum = net->bias2[i];
        for (int j = 0; j < 64; j++) {
            sum += hidden1[j] * net->weights2[j * 32 + i];
        }
        hidden2[i] = fmaxf(0.0f, sum);
    }
    
    // Output layer with softmax
    float logits[6];
    float maxLogit = -1e10f;
    for (int i = 0; i < 6; i++) {
        float sum = net->bias3[i];
        for (int j = 0; j < 32; j++) {
            sum += hidden2[j] * net->weights3[j * 6 + i];
        }
        logits[i] = sum;
        if (sum > maxLogit) maxLogit = sum;
    }
    
    // Softmax
    float sumExp = 0.0f;
    for (int i = 0; i < 6; i++) {
        probabilities[i] = expf(logits[i] - maxLogit);
        sumExp += probabilities[i];
    }
    for (int i = 0; i < 6; i++) {
        probabilities[i] /= sumExp;
    }
}

// ============================================================================
// RADAR MEASUREMENT SIMULATION (WITH NOISE)
// ============================================================================

struct RadarMeasurement {
    float range;               // Distance to target
    float rangeRate;           // Closing velocity
    float azimuth;             // Horizontal angle
    float elevation;           // Vertical angle
    float rcs;                 // Measured RCS
    float snr;                 // Signal-to-noise ratio
    Complex iqSample;          // Raw I/Q for micro-Doppler
    
    // Measurement uncertainties
    float rangeStd;
    float angleStd;
    float velocityStd;
};

__device__ inline RadarMeasurement simulateRadarMeasurement(
    float3 radarPos,
    float3 targetPos,
    float3 targetVel,
    TargetSignature* sig,
    Quaternion targetAtt,
    float time,
    curandState* randState  // For noise generation
) {
    RadarMeasurement meas;
    
    // True geometry
    float3 relPos = targetPos - radarPos;
    float3 relVel = targetVel;  // Assuming stationary radar
    
    float range = length(relPos);
    float3 los = relPos / range;
    
    meas.range = range;
    meas.rangeRate = -dot(relVel, los);
    meas.azimuth = atan2f(relPos.x, relPos.z);
    meas.elevation = asinf(relPos.y / range);
    
    // Calculate RCS
    Complex iq;
    meas.rcs = calculateRCS(sig, targetAtt, los, time, &iq);
    meas.iqSample = iq;
    
    // Signal-to-noise ratio (simplified radar equation)
    float Pt = 1e6f;     // Transmit power (W)
    float G = 1000.0f;   // Antenna gain
    float lambda = RADAR_WAVELENGTH;
    float k = 1.38e-23f; // Boltzmann
    float T = 290.0f;    // Noise temp (K)
    float B = 1e6f;      // Bandwidth (Hz)
    
    float Pr = (Pt * G * G * lambda * lambda * meas.rcs) / 
               (powf(4.0f * M_PI, 3) * powf(range, 4));
    float Pn = k * T * B;
    meas.snr = 10.0f * log10f(Pr / Pn);
    
    // Measurement uncertainties (depend on SNR)
    float snrLinear = powf(10.0f, meas.snr / 10.0f);
    meas.rangeStd = 15.0f / sqrtf(snrLinear);        // meters
    meas.angleStd = 0.001f / sqrtf(snrLinear);       // radians
    meas.velocityStd = 0.5f / sqrtf(snrLinear);      // m/s
    
    // Add measurement noise
    if (randState != nullptr) {
        meas.range += curand_normal(randState) * meas.rangeStd;
        meas.azimuth += curand_normal(randState) * meas.angleStd;
        meas.elevation += curand_normal(randState) * meas.angleStd;
        meas.rangeRate += curand_normal(randState) * meas.velocityStd;
    }
    
    return meas;
}

#endif // SENSOR_SIMULATION_CUH
