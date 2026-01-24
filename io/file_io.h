#ifndef FILE_IO_H
#define FILE_IO_H

#include "../core/globals.h"
#include "../simulation/cuda_utils.cuh" // For CUDA_CHECK

// Utility functions
inline void createDirectory(const char *path) {
  struct stat st = {0};
  if (stat(path, &st) == -1) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0700);
#endif
  }
}

inline void listModels() {
  printf("\nAvailable trained models:\n");
  printf("─────────────────────────────\n");

  int count = 0;

  try {
    for (const auto &entry : fs::directory_iterator(checkpointsDir)) {
      if (!entry.is_regular_file())
        continue;

      std::string name = entry.path().filename().string();

      if (name.find(".checkpoint") != std::string::npos) {
        std::string base = name.substr(0, name.size() - 11);
        printf("  [%d] %s\n", ++count, base.c_str());
      }
    }
  } catch (...) {
    printf("  Checkpoints directory not found.\n");
  }

  if (count == 0)
    printf("  No models found.\n");

  printf("─────────────────────────────\n\n");
}

inline bool loadCheckpoint(const char *modelName) {
  char filepath[512];
  sprintf(filepath, "%s/%s.checkpoint", checkpointsDir, modelName);

  FILE *f = fopen(filepath, "rb");
  if (!f) {
    printf("Could not load checkpoint: %s\n", filepath);
    return false;
  }

  fread(h_agent, sizeof(RLAgent), 1, f);
  fread(h_metrics, sizeof(TrainingMetrics), 1, f);
  fread(h_history, sizeof(TrainingHistory), 1, f);
  fclose(f);

  CUDA_CHECK(
      cudaMemcpy(d_agent, h_agent, sizeof(RLAgent), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metrics, h_metrics, sizeof(TrainingMetrics),
                        cudaMemcpyHostToDevice));

  printf("✓ Loaded checkpoint: %s\n", filepath);
  printf("  Episodes trained: %d\n", h_agent->episodeCount);
  printf("  Best success rate: %.2f%%\n", h_agent->bestSuccessRate * 100.0f);
  printf("  Total reward: %.2f\n", h_agent->totalReward);

  return true;
}

inline void saveCheckpoint() {
  char filepath[512];
  sprintf(filepath, "%s/%s_ep%d.checkpoint", checkpointsDir, modelName,
          h_agent->episodeCount);

  CUDA_CHECK(
      cudaMemcpy(h_agent, d_agent, sizeof(RLAgent), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, sizeof(TrainingMetrics),
                        cudaMemcpyDeviceToHost));

  FILE *f = fopen(filepath, "wb");
  if (f) {
    fwrite(h_agent, sizeof(RLAgent), 1, f);
    fwrite(h_metrics, sizeof(TrainingMetrics), 1, f);
    fwrite(h_history, sizeof(TrainingHistory), 1, f);
    fclose(f);
    printf("✓ Checkpoint saved: %s\n", filepath);
  }

  // Also save latest
  sprintf(filepath, "%s/%s.checkpoint", checkpointsDir, modelName);
  f = fopen(filepath, "wb");
  if (f) {
    fwrite(h_agent, sizeof(RLAgent), 1, f);
    fwrite(h_metrics, sizeof(TrainingMetrics), 1, f);
    fwrite(h_history, sizeof(TrainingHistory), 1, f);
    fclose(f);
  }
}

// ============================================================================
// COMPREHENSIVE CSV LOG INITIALIZATION
// ============================================================================
inline void initCSVLog() {
  FILE *f = fopen(csvFilePath, "w");
  if (f) {
    // Identification
    fprintf(f, "MissileID,Type,EpisodeID,");
    
    // Launch parameters
    fprintf(f, "LaunchTime,LaunchPosX,LaunchPosY,LaunchPosZ,");
    fprintf(f, "LaunchAngle,LaunchAzimuth,");
    fprintf(f, "InitVelX,InitVelY,InitVelZ,InitSpeed,");
    
    // Target information
    fprintf(f, "TargetX,TargetY,TargetZ,TargetInsideRadar,");
    
    // Impact data
    fprintf(f, "ImpactTime,ImpactPosX,ImpactPosY,ImpactPosZ,");
    fprintf(f, "ImpactVelX,ImpactVelY,ImpactVelZ,ImpactSpeed,ImpactAngle,");
    
    // Interception data
    fprintf(f, "Intercepted,InterceptorID,ResponseTime,DetectionTime,");
    fprintf(f, "InterceptionX,InterceptionY,InterceptionZ,InterceptionDistance,");
    
    // Trajectory statistics
    fprintf(f, "MaxAltitude,MaxVelocity,TotalDistance,FlightDuration,");
    fprintf(f, "FuelConsumed,AverageSpeed,AverageAcceleration,MaxAcceleration,");
    
    // PhysX aerodynamics data
    fprintf(f, "AvgDragForce,AvgLiftForce,AvgMachNumber,MaxMachNumber,");
    fprintf(f, "AvgDynamicPressure,AvgAirDensity,AvgTemperature,");
    
    // PINN prediction metrics
    fprintf(f, "PredictionError,TrajectoryDeviation,PINNConfidence,");
    fprintf(f, "PredictedImpactX,PredictedImpactY,PredictedImpactZ,");
    
    // Radar and defense metrics
    fprintf(f, "DistanceFromDefense,RadarCrossSection,TimeInRadar,");
    
    // RL reward data (only calculated if inside radar)
    fprintf(f, "RewardCalculated,TotalReward,");
    fprintf(f, "RewardIntercept,RewardSpeedBonus,RewardEarlyDetection,");
    fprintf(f, "RewardPredictionAccuracy,RewardFuelEfficiency,RewardDistanceBonus,");
    fprintf(f, "RewardComponent7,RewardComponent8,");
    
    // Agent state at time of event
    fprintf(f, "AgentEpsilon,AgentLearningRate,AgentEpisodeCount,");
    fprintf(f, "CurrentSuccessRate,MovingAvgReward,");
    
    // Training mode indicators
    fprintf(f, "TrainingMode,SimulationSpeed,GlobalTime\n");
    
    fclose(f);
  }
}

// ============================================================================
// LOG COMPREHENSIVE MISSILE EVENT
// ============================================================================
inline void logMissileEvent(const MissileEvent &event) {
  missileLog.push_back(event);

  FILE *f = fopen(csvFilePath, "a");
  if (f) {
    // Identification
    fprintf(f, "%d,%s,%d,",
            event.missileId,
            event.type == ENEMY_MISSILE ? "ENEMY" : "INTERCEPTOR",
            event.episodeId);
    
    // Launch parameters
    float initSpeed = sqrtf(event.initialVelocity.x * event.initialVelocity.x +
                            event.initialVelocity.y * event.initialVelocity.y +
                            event.initialVelocity.z * event.initialVelocity.z);
    fprintf(f, "%.4f,%.2f,%.2f,%.2f,",
            event.launchTime,
            event.launchPos.x, event.launchPos.y, event.launchPos.z);
    fprintf(f, "%.4f,%.4f,",
            event.launchAngle, event.launchAzimuth);
    fprintf(f, "%.2f,%.2f,%.2f,%.2f,",
            event.initialVelocity.x, event.initialVelocity.y,
            event.initialVelocity.z, initSpeed);
    
    // Target information
    fprintf(f, "%.2f,%.2f,%.2f,%d,",
            event.target.x, event.target.y, event.target.z,
            event.targetInsideRadar);
    
    // Impact data
    float impactSpeed = sqrtf(event.impactVelocity.x * event.impactVelocity.x +
                              event.impactVelocity.y * event.impactVelocity.y +
                              event.impactVelocity.z * event.impactVelocity.z);
    fprintf(f, "%.4f,%.2f,%.2f,%.2f,",
            event.impactTime,
            event.impactPos.x, event.impactPos.y, event.impactPos.z);
    fprintf(f, "%.2f,%.2f,%.2f,%.2f,%.4f,",
            event.impactVelocity.x, event.impactVelocity.y,
            event.impactVelocity.z, impactSpeed, event.impactAngle);
    
    // Interception data
    fprintf(f, "%d,%d,%.4f,%.4f,",
            event.intercepted, event.interceptorId,
            event.responseTime, event.detectionTime);
    fprintf(f, "%.2f,%.2f,%.2f,%.2f,",
            event.interceptionPoint.x, event.interceptionPoint.y,
            event.interceptionPoint.z, event.interceptionDistance);
    
    // Trajectory statistics
    fprintf(f, "%.2f,%.2f,%.2f,%.4f,",
            event.maxAltitude, event.maxVelocity,
            event.totalDistance, event.flightDuration);
    fprintf(f, "%.3f,%.2f,%.2f,%.2f,",
            event.fuelConsumed, event.averageSpeed,
            event.averageAcceleration, event.maxAcceleration);
    
    // PhysX aerodynamics data
    fprintf(f, "%.4f,%.4f,%.4f,%.4f,",
            event.averageDragForce, event.averageLiftForce,
            event.averageMachNumber, event.maxMachNumber);
    fprintf(f, "%.2f,%.6f,%.2f,",
            event.averageDynamicPressure,
            0.0f,  // avgAirDensity - calculated from altitude
            0.0f); // avgTemperature
    
    // PINN prediction metrics
    fprintf(f, "%.2f,%.2f,%.4f,",
            event.predictionError, event.trajectoryDeviation,
            event.pinnConfidence);
    fprintf(f, "%.2f,%.2f,%.2f,",
            0.0f, 0.0f, 0.0f);  // Predicted impact (filled during event)
    
    // Radar and defense metrics
    fprintf(f, "%.2f,%.4f,%.4f,",
            event.distanceFromDefense, event.radarCrossSection,
            0.0f);  // timeInRadar
    
    // RL reward data
    fprintf(f, "%d,%.4f,",
            event.rewardCalculated, event.reward);
    for (int i = 0; i < 8; i++) {
      fprintf(f, "%.4f,", event.rewardBreakdown[i]);
    }
    
    // Agent state (get from global)
    fprintf(f, "%.6f,%.8f,%d,",
            h_agent->epsilon, h_agent->learningRate, h_agent->episodeCount);
    
    float currentSuccessRate = (h_metrics->interceptSuccess + h_metrics->interceptFail > 0)
        ? (float)h_metrics->interceptSuccess / 
          (h_metrics->interceptSuccess + h_metrics->interceptFail)
        : 0.0f;
    fprintf(f, "%.4f,%.4f,",
            currentSuccessRate, h_agent->movingAvgReward);
    
    // Training mode indicators
    fprintf(f, "%d,%.1f,%.4f\n",
            trainingMode ? 1 : 0, simulationSpeed, globalTime);
    
    fclose(f);
  }
}

// ============================================================================
// LOG EPISODE METRICS
// ============================================================================
inline void logEpisode() {
  CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, sizeof(TrainingMetrics),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_agent, d_agent, sizeof(RLAgent), cudaMemcpyDeviceToHost));

  EpisodeMetrics ep;
  ep.episodeNum = h_agent->episodeCount;
  ep.interceptSuccess = h_metrics->interceptSuccess;
  ep.interceptFail = h_metrics->interceptFail;
  ep.totalLaunched = h_metrics->totalLaunched;
  ep.missilesInsideRadar = h_metrics->insideRadarCount;
  ep.missilesOutsideRadar = h_metrics->outsideRadarCount;
  ep.successRate = (ep.interceptSuccess + ep.interceptFail > 0)
                       ? (float)ep.interceptSuccess /
                             (ep.interceptSuccess + ep.interceptFail)
                       : 0.0f;
  ep.avgResponseTime = h_metrics->avgResponseTime;
  ep.avgInterceptDistance = h_metrics->avgInterceptDistance;
  ep.episodeReward = h_metrics->episodeReward;
  ep.epsilon = h_agent->epsilon;
  ep.learningRate = h_agent->learningRate;
  ep.timestamp = (double)(clock() - startTime) / CLOCKS_PER_SEC;
  
  // Advanced metrics
  ep.avgPredictionError = h_metrics->avgPredictionError;
  ep.avgMachNumber = h_metrics->avgMachNumber;
  ep.avgFuelEfficiency = 0.0f;  // Calculated elsewhere
  ep.avgTrajectoryDeviation = 0.0f;
  ep.ppoLoss = h_metrics->ppoLoss;
  ep.valueLoss = h_metrics->valueLoss;
  ep.entropyLoss = h_metrics->entropyLoss;
  ep.mamlMetaLoss = h_metrics->mamlMetaLoss;

  episodeLog.push_back(ep);

  // Update history for graphing
  if (h_history->count < MAX_HISTORY) {
    h_history->successRates[h_history->count] = ep.successRate;
    h_history->avgResponseTimes[h_history->count] = ep.avgResponseTime;
    h_history->rewards[h_history->count] = ep.episodeReward;
    h_history->predictionErrors[h_history->count] = ep.avgPredictionError;
    h_history->ppoLosses[h_history->count] = ep.ppoLoss;
    h_history->valueLosses[h_history->count] = ep.valueLoss;
    h_history->count++;
  } else {
    // Shift and add new
    for (int i = 0; i < MAX_HISTORY - 1; i++) {
      h_history->successRates[i] = h_history->successRates[i + 1];
      h_history->avgResponseTimes[i] = h_history->avgResponseTimes[i + 1];
      h_history->rewards[i] = h_history->rewards[i + 1];
      h_history->predictionErrors[i] = h_history->predictionErrors[i + 1];
      h_history->ppoLosses[i] = h_history->ppoLosses[i + 1];
      h_history->valueLosses[i] = h_history->valueLosses[i + 1];
    }
    h_history->successRates[MAX_HISTORY - 1] = ep.successRate;
    h_history->avgResponseTimes[MAX_HISTORY - 1] = ep.avgResponseTime;
    h_history->rewards[MAX_HISTORY - 1] = ep.episodeReward;
    h_history->predictionErrors[MAX_HISTORY - 1] = ep.avgPredictionError;
    h_history->ppoLosses[MAX_HISTORY - 1] = ep.ppoLoss;
    h_history->valueLosses[MAX_HISTORY - 1] = ep.valueLoss;
  }

  FILE *f = fopen(logFilePath, "a");
  if (f) {
    fprintf(f, "[Episode %d] Time: %.2fs | Success: %d/%d (%.1f%%) | ",
            ep.episodeNum, ep.timestamp, ep.interceptSuccess,
            ep.interceptSuccess + ep.interceptFail, ep.successRate * 100.0f);
    fprintf(f, "Inside/Outside Radar: %d/%d | ",
            ep.missilesInsideRadar, ep.missilesOutsideRadar);
    fprintf(f, "Reward: %.2f | Response: %.3fs | Epsilon: %.4f | ",
            ep.episodeReward, ep.avgResponseTime, ep.epsilon);
    fprintf(f, "PredErr: %.2f | Mach: %.2f | PPO: %.4f | MAML: %.4f\n",
            ep.avgPredictionError, ep.avgMachNumber, ep.ppoLoss, ep.mamlMetaLoss);
    fclose(f);
  }

  // Save checkpoint periodically
  if (h_agent->episodeCount % CHECKPOINT_INTERVAL == 0) {
    saveCheckpoint();
  }
}

#endif // FILE_IO_H
