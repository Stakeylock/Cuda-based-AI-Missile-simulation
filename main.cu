#include "core/config.h"
#include "core/globals.h"

// -- Definition of Global Variables --

// Station positions
float3 ENEMY_STATION = {-8000.0f, 0.0f, 0.0f};
float3 DEFENSE_STATION = {8000.0f, 0.0f, 0.0f};
__constant__ float3 d_DEFENSE_STATION;

// Global pointers
Missile *d_missiles = nullptr;
Missile *h_missiles = nullptr;
RLAgent *d_agent = nullptr;
RLAgent *h_agent = nullptr;
TrainingMetrics *d_metrics = nullptr;
TrainingMetrics *h_metrics = nullptr;
TrainingHistory *h_history = nullptr;

std::vector<MissileEvent> missileLog;
std::vector<EpisodeMetrics> episodeLog;

int enemyMissileCount = 0;
int interceptorCount = 0;
int totalMissileCount = 0;
int nextMissileId = 0;
bool trainingMode = false;
float simulationSpeed = 1.0f;
int trainingEpisodes = 0;
float globalTime = 0.0f;
int currentEpisodeId = 0;

// Model management
char modelName[256] = "default_model";
char logsDir[256] = "logs";
char checkpointsDir[256] = "checkpoints";
char csvFilePath[512];
char logFilePath[512];
bool newModel = true;

// Camera and UI
float cameraDistance = 25000.0f;
float cameraAngleX = 35.0f;
float cameraAngleY = 45.0f;
int mainWindow, graphWindow;
int windowWidth = 1600;
int windowHeight = 900;
int graphWidth = 800;
int graphHeight = 600;
int mouseX = 0, mouseY = 0;
bool mouseLeftDown = false;
bool mouseRightDown = false;
float fps = 0.0f;
int frameCount = 0;
clock_t lastTime;
clock_t startTime;

// Minimap
bool showMinimap = true;
float minimapX = 50.0f;
float minimapY = 50.0f;
float minimapSize = 250.0f;

// Mouse click targeting
float3 clickedTarget = {0.0f, 0.0f, 0.0f};
bool hasClickedTarget = false;

// Target marker visualization
bool showTargetMarker = true;
float targetMarkerPulse = 0.0f;

// Includes for other modules
#include "core/config.h"
#include "core/globals.h"
#include "io/file_io.h"
#include "render/renderer.h"
#include "simulation/cuda_utils.cuh"
#include "simulation/physx_engine.cuh"
#include "simulation/kernels.cuh"
#include "simulation/neural_net.cuh"

// ============================================================================
// HELPER: Screen to World coordinate conversion
// ============================================================================
float3 screenToWorld(int screenX, int screenY) {
    // Get OpenGL matrices
    GLdouble modelview[16], projection[16];
    GLint viewport[4];
    
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);
    
    // Read depth at click position
    float depth;
    glReadPixels(screenX, viewport[3] - screenY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    
    // Unproject to world coordinates
    GLdouble worldX, worldY, worldZ;
    gluUnProject((GLdouble)screenX, (GLdouble)(viewport[3] - screenY), depth,
                 modelview, projection, viewport,
                 &worldX, &worldY, &worldZ);
    
    // If clicked on sky (depth = 1.0), project to ground plane
    if (depth > 0.99f) {
        // Cast ray from camera through click point to ground plane
        GLdouble nearX, nearY, nearZ, farX, farY, farZ;
        gluUnProject(screenX, viewport[3] - screenY, 0.0,
                     modelview, projection, viewport,
                     &nearX, &nearY, &nearZ);
        gluUnProject(screenX, viewport[3] - screenY, 1.0,
                     modelview, projection, viewport,
                     &farX, &farY, &farZ);
        
        // Ray direction
        double dirX = farX - nearX;
        double dirY = farY - nearY;
        double dirZ = farZ - nearZ;
        
        // Find intersection with y = 0 plane
        if (fabs(dirY) > 0.001) {
            double t = -nearY / dirY;
            worldX = nearX + dirX * t;
            worldY = 0.0;
            worldZ = nearZ + dirZ * t;
        }
    }
    
    return make_float3((float)worldX, GROUND_LEVEL, (float)worldZ);
}

// ============================================================================
// GENERATE TARGET POSITION WITH 80/20 DISTRIBUTION
// ============================================================================
float3 generateTrainingTarget() {
    float randVal = (float)rand() / RAND_MAX;
    float3 target;
    
    if (randVal < INSIDE_RADAR_RATIO) {
        // 80% inside radar range - target near defense station
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float distance = ((float)rand() / RAND_MAX) * RADAR_RANGE * 0.9f;  // Within 90% of radar
        
        target.x = DEFENSE_STATION.x + distance * cosf(angle);
        target.y = GROUND_LEVEL;
        target.z = DEFENSE_STATION.z + distance * sinf(angle);
    } else {
        // 20% outside radar range
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float distance = RADAR_RANGE + ((float)rand() / RAND_MAX) * (WORLD_SIZE - RADAR_RANGE);
        
        target.x = DEFENSE_STATION.x + distance * cosf(angle);
        target.y = GROUND_LEVEL;
        target.z = DEFENSE_STATION.z + distance * sinf(angle);
        
        // Clamp to world bounds
        target.x = fmaxf(fminf(target.x, WORLD_SIZE - 1000.0f), -WORLD_SIZE + 1000.0f);
        target.z = fmaxf(fminf(target.z, WORLD_SIZE - 1000.0f), -WORLD_SIZE + 1000.0f);
    }
    
    return target;
}

// -- Implementation of Main Logic --

void initSimulation() {
  h_missiles = new Missile[MAX_MISSILES];
  h_agent = new RLAgent;
  h_metrics = new TrainingMetrics;
  h_history = new TrainingHistory;

  CUDA_CHECK(cudaMalloc(&d_missiles, MAX_MISSILES * sizeof(Missile)));
  CUDA_CHECK(cudaMalloc(&d_agent, sizeof(RLAgent)));
  CUDA_CHECK(cudaMalloc(&d_metrics, sizeof(TrainingMetrics)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_DEFENSE_STATION, &DEFENSE_STATION, sizeof(float3)));

  memset(h_missiles, 0, MAX_MISSILES * sizeof(Missile));
  memset(h_metrics, 0, sizeof(TrainingMetrics));
  memset(h_history, 0, sizeof(TrainingHistory));

  if (!newModel && loadCheckpoint(modelName)) {
    printf("✓ Continuing training from existing model\n");
  } else {
    printf("✓ Starting new model: %s\n", modelName);
    srand(time(NULL));
    
    // Initialize policy network
    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
      h_agent->policyNet.weights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
      h_agent->predictionNet.weights1[i] =
          ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }

    for (int i = 0; i < NEURAL_HIDDEN * NEURAL_HIDDEN; i++) {
      h_agent->policyNet.weights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
      h_agent->predictionNet.weights2[i] =
          ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    for (int i = 0; i < NEURAL_HIDDEN * 3; i++) {
      h_agent->policyNet.weights3[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
      h_agent->predictionNet.weights3[i] =
          ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Initialize Actor-Critic network for PPO
    for (int i = 0; i < 12 * NEURAL_HIDDEN; i++) {
      h_agent->actorCritic.sharedWeights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    for (int i = 0; i < NEURAL_HIDDEN * NEURAL_HIDDEN; i++) {
      h_agent->actorCritic.sharedWeights2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < NEURAL_HIDDEN * 6; i++) {
      h_agent->actorCritic.actorWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < 6; i++) {
      h_agent->actorCritic.actorLogStd[i] = -0.5f;  // Initial std ~ 0.6
    }
    for (int i = 0; i < NEURAL_HIDDEN; i++) {
      h_agent->actorCritic.criticWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Initialize SAC networks
    for (int i = 0; i < 6 * NEURAL_HIDDEN; i++) {
      h_agent->sacNets.qNetwork1.weights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
      h_agent->sacNets.qNetwork2.weights1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
      h_agent->sacNets.qTarget1.weights1[i] = h_agent->sacNets.qNetwork1.weights1[i];
      h_agent->sacNets.qTarget2.weights1[i] = h_agent->sacNets.qNetwork2.weights1[i];
    }
    h_agent->sacNets.logAlpha = 0.0f;
    h_agent->sacNets.targetEntropy = -6.0f;  // -dim(action)

    // Initialize MAML state
    memset(&h_agent->mamlState, 0, sizeof(MAMLState));

    // Initialize experience buffer
    h_agent->expBuffer.head = 0;
    h_agent->expBuffer.size = 0;
    h_agent->expBuffer.capacity = SAC_BUFFER_SIZE;

    // Initialize PPO buffer
    h_agent->ppoBuffer.count = 0;

    // Hyperparameters
    h_agent->learningRate = 0.001f;
    h_agent->epsilon = 0.3f;
    h_agent->gamma = 0.99f;
    h_agent->tau = SAC_TAU;
    h_agent->entropyCoef = PPO_ENTROPY_COEF;
    h_agent->valueCoef = PPO_VALUE_COEF;
    h_agent->clipEpsilon = PPO_CLIP_EPSILON;
    h_agent->gaeLambda = PPO_GAE_LAMBDA;
    
    h_agent->episodeCount = 0;
    h_agent->totalReward = 0.0f;
    h_agent->bestSuccessRate = 0.0f;
    h_agent->updateStep = 0;
    h_agent->trainingPhase = 0;
    h_agent->recentIndex = 0;
    h_agent->movingAvgReward = 0.0f;
    h_agent->movingAvgSuccessRate = 0.0f;

    CUDA_CHECK(
        cudaMemcpy(d_agent, h_agent, sizeof(RLAgent), cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMemcpy(d_missiles, h_missiles, MAX_MISSILES * sizeof(Missile),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metrics, h_metrics, sizeof(TrainingMetrics),
                        cudaMemcpyHostToDevice));

  lastTime = clock();
  startTime = clock();
}

void launchEnemyMissile(float targetX, float targetZ) {
  if (totalMissileCount >= MAX_MISSILES - 100)
    return;

  float3 startPos = ENEMY_STATION;
  float3 targetPos = make_float3(targetX, GROUND_LEVEL, targetZ);

  float3 flatDir =
      make_float3(targetX - startPos.x, 0.0f, targetZ - startPos.z);
  float distance = sqrtf(flatDir.x * flatDir.x + flatDir.z * flatDir.z);

  if (distance < 1.0f)
    return;

  // ========================================================================
  // USE INVERSE BALLISTIC TRAJECTORY SOLVER
  // Compute optimal launch angle and velocity to hit target
  // ========================================================================
  
  // Determine if we should use random angle for training diversity
  float desiredAngle = -1.0f;  // -1 means use optimal solver
  
  if (trainingMode) {
    // Training mode: use diverse angles (15° to 75°)
    // This ensures the AI learns to handle various trajectory profiles
    float randVal = (float)rand() / RAND_MAX;
    desiredAngle = 0.26f + randVal * 1.05f;  // ~15° to ~75°
  }
  
  InverseTrajectoryResult trajectory;

  float minSpeed = 200.0f;
  float maxSpeed = fminf(2000.0f, 300.0f + distance * 0.12f);
  if (maxSpeed < minSpeed + 50.0f) {
    maxSpeed = minSpeed + 50.0f;
  }
  
  if (desiredAngle > 0.0f) {
    // Training: use fixed angle, solve for speed
    trajectory = solveTrajectoryWithRandomAngle(
      startPos, targetPos,
      MISSILE_MASS, DRAG_COEFFICIENT, REFERENCE_AREA_MISSILE,
      0.0f, 0.0f,
      desiredAngle,
      minSpeed,
      maxSpeed
    );
  } else {
    // Operational: use optimal inverse trajectory
    trajectory = solveInverseTrajectory(
      startPos, targetPos,
      MISSILE_MASS, DRAG_COEFFICIENT, REFERENCE_AREA_MISSILE,
      0.0f, 0.0f,
      minSpeed,
      maxSpeed
    );
  }
  
  // Fallback to simple ballistic if solver fails
  float3 initialVel;
  float launchAngle, launchSpeed;
  
  if (trajectory.valid && trajectory.finalError < 1500.0f) {
    initialVel = trajectory.initialVelocity;
    launchAngle = trajectory.launchAngle;
    launchSpeed = trajectory.launchSpeed;
  } else {
    // Fallback: analytic ballistic with drag compensation
    float3 horizDir = make_float3(flatDir.x / distance, 0.0f, flatDir.z / distance);
    launchAngle = (desiredAngle > 0.0f) ? desiredAngle : (M_PI / 4.0f);
    float sin2 = sinf(2.0f * launchAngle);
    if (fabsf(sin2) < 1e-3f) sin2 = 1e-3f;
    float baseSpeed = sqrtf(fmaxf(0.0f, distance * GRAVITY / sin2));
    float dragComp = 1.0f + fminf(0.6f, distance * 0.00002f);
    launchSpeed = fminf(maxSpeed, fmaxf(minSpeed, baseSpeed * dragComp));
    float vxz = launchSpeed * cosf(launchAngle);
    float vy = launchSpeed * sinf(launchAngle);
    initialVel = make_float3(horizDir.x * vxz, vy, horizDir.z * vxz);
  }

  // Check if target is inside radar
  float3 toTarget = targetPos - DEFENSE_STATION;
  bool targetInsideRadar = sqrtf(toTarget.x * toTarget.x + toTarget.z * toTarget.z) < RADAR_RANGE;

  Missile m;
  memset(&m, 0, sizeof(Missile));
  
  // Core state
  m.position = startPos;
  m.prevPosition = startPos;
  m.velocity = initialVel;
  m.prevVelocity = initialVel;
  m.initialVelocity = initialVel;
  m.acceleration = make_float3(0.0f, 0.0f, 0.0f);
  m.angularVelocity = make_float3(0.0f, 0.0f, SPIN_RATE);
  m.orientation = make_float3(0.0f, 0.0f, 0.0f);
  m.force = make_float3(0.0f, 0.0f, 0.0f);
  m.torque = make_float3(0.0f, 0.0f, 0.0f);
  
  // Target
  m.target = targetPos;
  m.launchPos = startPos;
  m.predictedImpact = targetPos;
  
  // Physical properties
  m.fuel = 0.0f;
  m.mass = MISSILE_MASS;
  m.dragCoefficient = DRAG_COEFFICIENT;
  m.liftCoefficient = LIFT_COEFFICIENT;
  m.referenceArea = REFERENCE_AREA_MISSILE;
  m.machNumber = 0.0f;
  m.dynamicPressure = 0.0f;
  m.altitude = startPos.y;
  m.airDensity = AIR_DENSITY;
  m.temperature = SEA_LEVEL_TEMP;
  
  // State flags
  m.active = 1;
  m.hit = 0;
  m.type = ENEMY_MISSILE;
  m.insideRadar = 0;
  m.targetInsideRadar = targetInsideRadar ? 1 : 0;
  
  // Timing
  m.lifetime = 0.0f;
  m.launchTime = globalTime;
  m.detectionTime = 0.0f;
  m.interceptTime = 0.0f;
  
  // IDs
  m.targetMissileId = -1;
  m.missileId = nextMissileId++;
  
  // Statistics
  m.maxAltitude = 0.0f;
  m.maxVelocity = launchSpeed;
  m.distanceTraveled = 0.0f;
  m.fuelConsumed = 0.0f;
  m.averageThrust = 0.0f;
  m.maxAcceleration = 0.0f;
  m.minDistanceToTarget = distance;
  m.predictionError = 0.0f;
  m.substepAccumulator = 0.0f;

  h_missiles[totalMissileCount] = m;
  CUDA_CHECK(cudaMemcpy(&d_missiles[totalMissileCount], &m, sizeof(Missile),
                        cudaMemcpyHostToDevice));
  totalMissileCount++;
  enemyMissileCount++;
  h_metrics->totalMissilesFired++;
}

void updateSimulation(float dt) {
  if (totalMissileCount == 0)
    return;

  dt *= simulationSpeed;
  globalTime += dt;
  int blocks = (totalMissileCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int *d_interceptorCount;
  CUDA_CHECK(cudaMalloc(&d_interceptorCount, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_interceptorCount, &interceptorCount, sizeof(int),
                        cudaMemcpyHostToDevice));

  radarDetectionKernel<<<blocks, BLOCK_SIZE>>>(
      d_missiles, totalMissileCount, d_agent, d_metrics, d_interceptorCount, dt,
      globalTime);

  CUDA_CHECK(cudaMemcpy(&interceptorCount, d_interceptorCount, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_interceptorCount));

  totalMissileCount = std::max(totalMissileCount, interceptorCount);

  updateMissilesKernel<<<blocks, BLOCK_SIZE>>>(d_missiles, totalMissileCount,
                                               dt, d_metrics);

  if (trainingMode) {
    updateRLAgentKernel<<<1, 1>>>(d_agent, d_metrics, dt);
    trainingEpisodes++;

    if (trainingEpisodes % 50 == 0) {
      logEpisode();
      currentEpisodeId++;

      memset(h_metrics, 0, sizeof(TrainingMetrics));
      CUDA_CHECK(cudaMemcpy(d_metrics, h_metrics, sizeof(TrainingMetrics),
                            cudaMemcpyHostToDevice));
    }

    // Launch missiles with 80/20 distribution (inside/outside radar)
    if (trainingEpisodes % 30 == 0) {
      for (int i = 0; i < 5; i++) {
        float3 target = generateTrainingTarget();
        launchEnemyMissile(target.x, target.z);
      }
    }
  }

  CUDA_CHECK(cudaMemcpy(h_missiles, d_missiles,
                        totalMissileCount * sizeof(Missile),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < totalMissileCount; i++) {
    if (!h_missiles[i].active && h_missiles[i].hit &&
        h_missiles[i].lifetime > 0.1f) {
      bool alreadyLogged = false;
      for (const auto &log : missileLog) {
        if (log.missileId == h_missiles[i].missileId) {
          alreadyLogged = true;
          break;
        }
      }

      if (!alreadyLogged) {
        MissileEvent event;
        memset(&event, 0, sizeof(MissileEvent));
        
        // Identification
        event.missileId = h_missiles[i].missileId;
        event.type = h_missiles[i].type;
        event.episodeId = currentEpisodeId;
        
        // Launch data
        event.launchPos = h_missiles[i].launchPos;
        event.target = h_missiles[i].target;
        event.initialVelocity = h_missiles[i].initialVelocity;
        event.launchTime = h_missiles[i].launchTime;
        
        // Calculate launch angle and azimuth
        float3 launchDir = h_missiles[i].initialVelocity;
        float horizontalSpeed = sqrtf(launchDir.x * launchDir.x + launchDir.z * launchDir.z);
        event.launchAngle = atan2f(launchDir.y, horizontalSpeed);
        event.launchAzimuth = atan2f(launchDir.x, launchDir.z);
        
        // Impact data
        event.impactTime = globalTime;
        event.impactPos = h_missiles[i].position;
        event.impactVelocity = h_missiles[i].velocity;
        event.impactAngle = atan2f(-h_missiles[i].velocity.y, 
            sqrtf(h_missiles[i].velocity.x * h_missiles[i].velocity.x + 
                  h_missiles[i].velocity.z * h_missiles[i].velocity.z));
        
        // Interception data
        event.intercepted = (h_missiles[i].type == ENEMY_MISSILE && h_missiles[i].hit);
        event.interceptorId = h_missiles[i].targetMissileId;
        event.responseTime = h_missiles[i].lifetime;
        event.detectionTime = h_missiles[i].detectionTime;
        event.interceptionPoint = h_missiles[i].position;
        event.interceptionDistance = h_missiles[i].minDistanceToTarget;
        
        // Trajectory statistics
        event.maxAltitude = h_missiles[i].maxAltitude;
        event.maxVelocity = h_missiles[i].maxVelocity;
        event.totalDistance = h_missiles[i].distanceTraveled;
        event.flightDuration = h_missiles[i].lifetime;
        event.fuelConsumed = h_missiles[i].fuelConsumed;
        event.averageSpeed = h_missiles[i].distanceTraveled / fmaxf(h_missiles[i].lifetime, 0.01f);
        event.averageAcceleration = h_missiles[i].averageThrust / h_missiles[i].mass;
        event.maxAcceleration = h_missiles[i].maxAcceleration;
        
        // PhysX data
        event.averageDragForce = 0.0f;  // Accumulated in simulation
        event.averageLiftForce = 0.0f;
        event.averageMachNumber = h_missiles[i].machNumber;
        event.maxMachNumber = h_missiles[i].machNumber;
        event.averageDynamicPressure = h_missiles[i].dynamicPressure;
        
        // PINN metrics
        event.predictionError = h_missiles[i].predictionError;
        event.trajectoryDeviation = 0.0f;
        event.pinnConfidence = 0.8f;
        
        // Radar data
        event.targetInsideRadar = h_missiles[i].targetInsideRadar;
        float3 toDefense = DEFENSE_STATION - h_missiles[i].position;
        event.distanceFromDefense = sqrtf(toDefense.x * toDefense.x + toDefense.z * toDefense.z);
        event.radarCrossSection = h_missiles[i].referenceArea;
        
        // Reward (only if inside radar)
        event.rewardCalculated = h_missiles[i].targetInsideRadar;
        if (event.rewardCalculated) {
            event.reward = h_missiles[i].type == INTERCEPTOR_MISSILE ? 
                REWARD_INTERCEPT_SUCCESS - h_missiles[i].lifetime * 2.0f : 0.0f;
        }
        for (int r = 0; r < 8; r++) event.rewardBreakdown[r] = 0.0f;

        logMissileEvent(event);
      }
    }
  }

  CUDA_CHECK(cudaGetLastError());
}

// Display Wrapper
void displayMain() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (double)windowWidth / (double)windowHeight, 10.0,
                 100000.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float camX = cameraDistance * sinf(cameraAngleY * M_PI / 180.0f) *
               cosf(cameraAngleX * M_PI / 180.0f);
  float camY = cameraDistance * sinf(cameraAngleX * M_PI / 180.0f);
  float camZ = cameraDistance * cosf(cameraAngleY * M_PI / 180.0f) *
               cosf(cameraAngleX * M_PI / 180.0f);

  gluLookAt(camX, camY, camZ, 0, 500, 0, 0, 1, 0);

  drawGrid();
  drawStation(ENEMY_STATION, 1.0f, 0.0f, 0.0f);
  drawStation(DEFENSE_STATION, 0.0f, 1.0f, 0.0f);
  drawMissiles();
  
  // Draw target marker in 3D view
  if (hasClickedTarget && showTargetMarker) {
    drawTargetMarker3D(clickedTarget, targetMarkerPulse);
  }
  
  drawMinimap();
  
  // Draw target marker on minimap (needs 2D context)
  if (hasClickedTarget && showTargetMarker && showMinimap) {
    // Set up 2D projection for minimap marker
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowWidth, 0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    
    drawTargetMarkerMinimap(clickedTarget, targetMarkerPulse);
    
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
  }
  
  drawHUD();

  glutSwapBuffers();

  frameCount++;
  clock_t currentTime = clock();
  double elapsed = (double)(currentTime - lastTime) / CLOCKS_PER_SEC;
  if (elapsed >= 1.0) {
    fps = frameCount / elapsed;
    frameCount = 0;
    lastTime = currentTime;
  }
}

void idle() {
  static clock_t lastFrame = clock();
  clock_t currentFrame = clock();
  float dt = (float)(currentFrame - lastFrame) / CLOCKS_PER_SEC;
  lastFrame = currentFrame;

  if (dt > 0.1f)
    dt = 0.016f;

  // Update target marker pulse animation
  targetMarkerPulse += dt * 2.0f;
  if (targetMarkerPulse > 2.0f * M_PI) {
    targetMarkerPulse -= 2.0f * M_PI;
  }

  updateSimulation(dt);

  glutSetWindow(mainWindow);
  glutPostRedisplay();

  glutSetWindow(graphWindow);
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    // Check if clicking on minimap first
    if (showMinimap && x >= minimapX && x <= minimapX + minimapSize &&
        (windowHeight - y) >= minimapY &&
        (windowHeight - y) <= minimapY + minimapSize) {

      float relX = (x - minimapX - minimapSize / 2.0f) / (minimapSize / 2.0f);
      float relY = ((windowHeight - y) - minimapY - minimapSize / 2.0f) /
                   (minimapSize / 2.0f);

      float targetX = relX * WORLD_SIZE;
      float targetZ = relY * WORLD_SIZE;

      // Set target marker for minimap clicks too
      clickedTarget = make_float3(targetX, GROUND_LEVEL, targetZ);
      hasClickedTarget = true;
      targetMarkerPulse = 0.0f;  // Reset pulse animation

      launchEnemyMissile(targetX, targetZ);
      
      // Check if inside radar
      float3 toDefense = clickedTarget - DEFENSE_STATION;
      float distToDefense = sqrtf(toDefense.x * toDefense.x + toDefense.z * toDefense.z);
      bool insideRadar = distToDefense < RADAR_RANGE;
      
      printf("Missile launched at target: (%.1f, %.1f) via minimap [%s radar]\n", 
             targetX, targetZ, insideRadar ? "INSIDE" : "OUTSIDE");
    } else {
      // Click on 3D view - convert to world coordinates
      float3 worldPos = screenToWorld(x, y);
      
      // Clamp to world bounds
      worldPos.x = fmaxf(fminf(worldPos.x, WORLD_SIZE - 100.0f), -WORLD_SIZE + 100.0f);
      worldPos.z = fmaxf(fminf(worldPos.z, WORLD_SIZE - 100.0f), -WORLD_SIZE + 100.0f);
      
      // Store clicked target and launch missile
      clickedTarget = worldPos;
      hasClickedTarget = true;
      targetMarkerPulse = 0.0f;  // Reset pulse animation
      
      launchEnemyMissile(worldPos.x, worldPos.z);
      
      // Check if inside radar
      float3 toDefense = worldPos - DEFENSE_STATION;
      float distToDefense = sqrtf(toDefense.x * toDefense.x + toDefense.z * toDefense.z);
      bool insideRadar = distToDefense < RADAR_RANGE;
      
      printf("Missile launched at target: (%.1f, %.1f) via 3D click [%s radar]\n", 
             worldPos.x, worldPos.z, insideRadar ? "INSIDE" : "OUTSIDE");
    }
  }

  if (button == GLUT_RIGHT_BUTTON) {
    mouseRightDown = (state == GLUT_DOWN);
    if (mouseRightDown) {
      mouseX = x;
      mouseY = y;
    }
  }
}

void motion(int x, int y) {
  if (mouseRightDown) {
    int dx = x - mouseX;
    int dy = y - mouseY;

    cameraAngleY += dx * 0.5f;
    cameraAngleX += dy * 0.5f;

    if (cameraAngleX > 89.0f)
      cameraAngleX = 89.0f;
    if (cameraAngleX < -89.0f)
      cameraAngleX = -89.0f;

    mouseX = x;
    mouseY = y;
  }
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case 27:
    saveCheckpoint();
    printf("\n✓ Final checkpoint saved. Exiting...\n");
    exit(0);
    break;
  case 't':
  case 'T':
    trainingMode = !trainingMode;
    simulationSpeed = trainingMode ? TRAINING_SPEED : 1.0f;
    printf("Training mode: %s (Speed: %.0fx)\n", trainingMode ? "ON" : "OFF",
           simulationSpeed);
    break;
  case 's':
  case 'S':
    saveCheckpoint();
    printf("Manual checkpoint saved!\n");
    break;
  case 'r':
  case 'R':
    totalMissileCount = 0;
    enemyMissileCount = 0;
    interceptorCount = 0;
    memset(h_missiles, 0, MAX_MISSILES * sizeof(Missile));
    memset(h_metrics, 0, sizeof(TrainingMetrics));
    CUDA_CHECK(cudaMemcpy(d_missiles, h_missiles,
                          MAX_MISSILES * sizeof(Missile),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metrics, h_metrics, sizeof(TrainingMetrics),
                          cudaMemcpyHostToDevice));
    printf("Simulation reset\n");
    break;
  case 'm':
  case 'M':
    showMinimap = !showMinimap;
    break;
  case ' ':
    // Launch missiles with 80/20 distribution
    for (int i = 0; i < 10; i++) {
      float3 target = generateTrainingTarget();
      launchEnemyMissile(target.x, target.z);
    }
    printf("Launched 10 missiles (80%% inside, 20%% outside radar)\n");
    break;
  }
}

void reshape(int w, int h) {
  windowWidth = w;
  windowHeight = h;
  glViewport(0, 0, w, h);
}

void reshapeGraph(int w, int h) {
  graphWidth = w;
  graphHeight = h;
  glViewport(0, 0, w, h);
}

void cleanup() {
  saveCheckpoint();

  if (d_missiles)
    cudaFree(d_missiles);
  if (d_agent)
    cudaFree(d_agent);
  if (d_metrics)
    cudaFree(d_metrics);
  if (h_missiles)
    delete[] h_missiles;
  if (h_agent)
    delete h_agent;
  if (h_metrics)
    delete h_metrics;
  if (h_history)
    delete h_history;

  printf("\n=== Training Summary ===\n");
  printf("Total episodes: %llu\n", (unsigned long long)episodeLog.size());
  printf("Total missile events logged: %llu\n",
         (unsigned long long)missileLog.size());
  printf("Logs saved to: %s\n", logsDir);
  printf("Checkpoints saved to: %s\n", checkpointsDir);
}

void getUserInput() {
  printf("\n╔═══════════════════════════════════════════════════════════╗\n");
  printf("║       AI MISSILE DEFENSE - MODEL CONFIGURATION           ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");

  createDirectory(logsDir);
  createDirectory(checkpointsDir);

  printf("Choose an option:\n");
  printf("  [1] Start new model\n");
  printf("  [2] Continue existing model\n");
  printf("\nEnter choice (1 or 2): ");

  int choice;
  if (scanf("%d", &choice) != 1) {
    choice = 1;
  }

  if (choice == 2) {
    listModels();
    printf("Enter model name to load: ");
    scanf("%s", modelName);
    newModel = false;
  } else {
    printf("Enter new model name: ");
    scanf("%s", modelName);
    newModel = true;
  }

  sprintf(csvFilePath, "%s/%s_missiles.csv", logsDir, modelName);
  sprintf(logFilePath, "%s/%s_training.log", logsDir, modelName);

  if (newModel) {
    initCSVLog();
    FILE *f = fopen(logFilePath, "w");
    if (f) {
      fprintf(f, "=== AI Missile Defense Training Log ===\n");
      time_t now = time(nullptr);
      fprintf(f, "Model: %s\n", modelName);
      fprintf(f, "Started: %s\n", ctime(&now));
      fprintf(f, "========================================\n\n");
      fclose(f);
    }
  } else {
    FILE *f = fopen(logFilePath, "a");
    if (f) {
      fprintf(f, "\n=== Training Resumed ===\n");
      time_t now = time(nullptr);
      fprintf(f, "Time: %s\n", ctime(&now));
      fclose(f);
    }
  }

  printf("\n✓ Configuration complete!\n");
  printf("  Model: %s\n", modelName);
  printf("  CSV Log: %s\n", csvFilePath);
  printf("  Training Log: %s\n", logFilePath);
  printf("  Checkpoints: %s/%s_*.checkpoint\n\n", checkpointsDir, modelName);
}

int main(int argc, char **argv) {
  getUserInput();

  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║   AI-POWERED MISSILE DEFENSE WITH RL & CHECKPOINTING     ║\n");
  printf("║        NVIDIA PhysX-Style Physics + PINN + PPO/SAC/MAML  ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");
  printf("FEATURES:\n");
  printf("  ✓ NVIDIA PhysX-style physics simulation (substeps, drag, lift)\n");
  printf("  ✓ Physics-Informed Neural Networks (PINN) for trajectory prediction\n");
  printf("  ✓ Advanced RL: PPO, SAC, and MAML meta-learning\n");
  printf("  ✓ Training: 80%% targets inside radar, 20%% outside\n");
  printf("  ✓ Rewards calculated ONLY for targets inside radar\n");
  printf("  ✓ Automatic checkpoint saving (every %d episodes)\n",
         CHECKPOINT_INTERVAL);
  printf("  ✓ Comprehensive CSV logging (60+ parameters)\n");
  printf("  ✓ Real-time training metrics & graphs\n");
  printf("  ✓ Model persistence & loading\n\n");
  printf("CONTROLS:\n");
  printf("  Left Click (3D)  : Set target & launch enemy missile\n");
  printf("  Left Click (Map) : Launch enemy missile at map position\n");
  printf("  Right Drag       : Rotate camera\n");
  printf("  T                : Toggle training mode (1x/100x speed)\n");
  printf("  S                : Save checkpoint manually\n");
  printf("  SPACE            : Launch 10 test missiles (80/20 distribution)\n");
  printf("  M                : Toggle minimap\n");
  printf("  R                : Reset simulation\n");
  printf("  ESC              : Save & Exit\n\n");
  printf("LOGGING:\n");
  printf("  Missile CSV      : %s\n", csvFilePath);
  printf("  Training Log     : %s\n", logFilePath);
  printf("  Checkpoints      : %s/\n\n", checkpointsDir);

  srand(time(NULL));

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

  // Main 3D window
  glutInitWindowSize(windowWidth, windowHeight);
  glutInitWindowPosition(50, 50);
  mainWindow = glutCreateWindow("AI Missile Defense - 3D View");

  glEnable(GL_DEPTH_TEST);
  glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
  glEnable(GL_POINT_SMOOTH);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

  glutDisplayFunc(displayMain);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);
  glutReshapeFunc(reshape);

  // Graph window
  glutInitWindowSize(graphWidth, graphHeight);
  glutInitWindowPosition(50 + windowWidth + 20, 50);
  graphWindow = glutCreateWindow("Training Metrics - Real-time");

  glClearColor(0.05f, 0.05f, 0.1f, 1.0f);

  glutDisplayFunc(displayGraph);
  glutReshapeFunc(reshapeGraph);

  // Initialize simulation
  glutSetWindow(mainWindow);
  initSimulation();

  // Idle function for both windows
  glutIdleFunc(idle);

  atexit(cleanup);

  printf("Initialization complete. System ready.\n");
  printf("Starting simulation...\n\n");

  glutMainLoop();
  return 0;
}