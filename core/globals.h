#ifndef GLOBALS_H
#define GLOBALS_H

#include "config.h"

// Station positions
extern float3 ENEMY_STATION;
extern float3 DEFENSE_STATION;
// extern __constant__ float3 d_DEFENSE_STATION; // Removed to avoid linkage
// issues in single-TU build

// Global variables
extern Missile *d_missiles;
extern Missile *h_missiles;
extern RLAgent *d_agent;
extern RLAgent *h_agent;
extern TrainingMetrics *d_metrics;
extern TrainingMetrics *h_metrics;
extern TrainingHistory *h_history;

extern std::vector<MissileEvent> missileLog;
extern std::vector<EpisodeMetrics> episodeLog;

extern int enemyMissileCount;
extern int interceptorCount;
extern int totalMissileCount;
extern int nextMissileId;
extern bool trainingMode;
extern float simulationSpeed;
extern int trainingEpisodes;
extern float globalTime;
extern int currentEpisodeId;

// Model management
extern char modelName[256];
extern char logsDir[256];
extern char checkpointsDir[256];
extern char csvFilePath[512];
extern char logFilePath[512];
extern bool newModel;

// Camera and UI
extern float cameraDistance;
extern float cameraAngleX;
extern float cameraAngleY;
extern int mainWindow, graphWindow;
extern int windowWidth;
extern int windowHeight;
extern int graphWidth;
extern int graphHeight;
extern int mouseX, mouseY;
extern bool mouseLeftDown;
extern bool mouseRightDown;
extern float fps;
extern int frameCount;
extern clock_t lastTime;
extern clock_t startTime;

// Minimap
extern bool showMinimap;
extern float minimapX;
extern float minimapY;
extern float minimapSize;

// Mouse click targeting
extern float3 clickedTarget;
extern bool hasClickedTarget;

// Target marker visualization
extern bool showTargetMarker;
extern float targetMarkerPulse;  // For animated pulse effect

#endif // GLOBALS_H
