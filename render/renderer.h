#ifndef RENDERER_H
#define RENDERER_H

#include "../core/globals.h"
#include "../simulation/cuda_utils.cuh"

inline void drawStation(float3 pos, float r, float g, float b) {
  glColor3f(r, g, b);
  glPushMatrix();
  glTranslatef(pos.x, pos.y + 100.0f, pos.z);
  glutSolidSphere(150.0f, 16, 16);
  glPopMatrix();

  glColor4f(r, g, b, 0.3f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 64; i++) {
    float angle = i * 2.0f * M_PI / 64.0f;
    glVertex3f(pos.x + RADAR_RANGE * cosf(angle), pos.y + 10.0f,
               pos.z + RADAR_RANGE * sinf(angle));
  }
  glEnd();
  glDisable(GL_BLEND);
}

inline void drawGrid() {
  glColor3f(0.15f, 0.15f, 0.2f);
  glBegin(GL_LINES);
  for (float i = -WORLD_SIZE; i <= WORLD_SIZE; i += 2000.0f) {
    glVertex3f(i, GROUND_LEVEL, -WORLD_SIZE);
    glVertex3f(i, GROUND_LEVEL, WORLD_SIZE);
    glVertex3f(-WORLD_SIZE, GROUND_LEVEL, i);
    glVertex3f(WORLD_SIZE, GROUND_LEVEL, i);
  }
  glEnd();
}

inline void drawMissiles() {
  glPointSize(4.0f);
  glBegin(GL_POINTS);

  for (int i = 0; i < totalMissileCount; i++) {
    if (!h_missiles[i].active)
      continue;

    if (h_missiles[i].type == ENEMY_MISSILE) {
      glColor3f(1.0f, 0.2f, 0.0f);
    } else {
      glColor3f(0.0f, 0.8f, 1.0f);
    }

    glVertex3f(h_missiles[i].position.x, h_missiles[i].position.y,
               h_missiles[i].position.z);
  }

  glEnd();
}

// ============================================================================
// TARGET MARKER VISUALIZATION
// Renders a red X marker at the clicked target location in 3D space
// ============================================================================
inline void drawTargetMarker3D(float3 target, float pulse) {
  if (!hasClickedTarget) return;
  
  // Animated pulse effect (0.0 to 1.0 cycling)
  float scale = 150.0f + 50.0f * sinf(pulse * 6.0f);  // Pulsing size
  float alpha = 0.7f + 0.3f * sinf(pulse * 4.0f);    // Pulsing alpha
  
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glLineWidth(3.0f);
  
  // Draw red X marker
  glColor4f(1.0f, 0.0f, 0.0f, alpha);
  
  // First line of X (NE to SW)
  glBegin(GL_LINES);
  glVertex3f(target.x - scale, GROUND_LEVEL + 10.0f, target.z - scale);
  glVertex3f(target.x + scale, GROUND_LEVEL + 10.0f, target.z + scale);
  glEnd();
  
  // Second line of X (NW to SE)
  glBegin(GL_LINES);
  glVertex3f(target.x - scale, GROUND_LEVEL + 10.0f, target.z + scale);
  glVertex3f(target.x + scale, GROUND_LEVEL + 10.0f, target.z - scale);
  glEnd();
  
  // Draw vertical marker pole
  glColor4f(1.0f, 0.2f, 0.2f, alpha * 0.6f);
  glBegin(GL_LINES);
  glVertex3f(target.x, GROUND_LEVEL, target.z);
  glVertex3f(target.x, GROUND_LEVEL + 500.0f, target.z);
  glEnd();
  
  // Draw circular ground indicator
  glColor4f(1.0f, 0.0f, 0.0f, alpha * 0.4f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float angle = i * 2.0f * M_PI / 32.0f;
    glVertex3f(target.x + scale * cosf(angle), GROUND_LEVEL + 5.0f, 
               target.z + scale * sinf(angle));
  }
  glEnd();
  
  // Draw outer expanding ring
  float outerScale = scale * (1.5f + 0.5f * sinf(pulse * 3.0f));
  glColor4f(1.0f, 0.3f, 0.0f, alpha * 0.3f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float angle = i * 2.0f * M_PI / 32.0f;
    glVertex3f(target.x + outerScale * cosf(angle), GROUND_LEVEL + 5.0f, 
               target.z + outerScale * sinf(angle));
  }
  glEnd();
  
  // Draw crosshair at center
  glColor4f(1.0f, 1.0f, 0.0f, alpha);
  float crossSize = scale * 0.3f;
  glBegin(GL_LINES);
  // Horizontal
  glVertex3f(target.x - crossSize, GROUND_LEVEL + 15.0f, target.z);
  glVertex3f(target.x + crossSize, GROUND_LEVEL + 15.0f, target.z);
  // Vertical (in z direction)
  glVertex3f(target.x, GROUND_LEVEL + 15.0f, target.z - crossSize);
  glVertex3f(target.x, GROUND_LEVEL + 15.0f, target.z + crossSize);
  glEnd();
  
  glLineWidth(1.0f);
  glDisable(GL_BLEND);
}

// ============================================================================
// TARGET MARKER ON MINIMAP
// Renders a red X marker at the clicked target location on the minimap
// ============================================================================
inline void drawTargetMarkerMinimap(float3 target, float pulse) {
  if (!hasClickedTarget || !showMinimap) return;
  
  float scaleX = minimapSize / (2.0f * WORLD_SIZE);
  float scaleZ = minimapSize / (2.0f * WORLD_SIZE);
  float centerX = minimapX + minimapSize / 2.0f;
  float centerY = minimapY + minimapSize / 2.0f;
  
  float tx = centerX + target.x * scaleX;
  float ty = centerY + target.z * scaleZ;
  
  // Check if target is within minimap bounds
  if (tx < minimapX || tx > minimapX + minimapSize ||
      ty < minimapY || ty > minimapY + minimapSize) return;
  
  float alpha = 0.8f + 0.2f * sinf(pulse * 4.0f);
  float markerSize = 8.0f + 3.0f * sinf(pulse * 6.0f);
  
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glLineWidth(2.0f);
  
  // Draw red X
  glColor4f(1.0f, 0.0f, 0.0f, alpha);
  glBegin(GL_LINES);
  // First line of X
  glVertex2f(tx - markerSize, ty - markerSize);
  glVertex2f(tx + markerSize, ty + markerSize);
  // Second line of X
  glVertex2f(tx - markerSize, ty + markerSize);
  glVertex2f(tx + markerSize, ty - markerSize);
  glEnd();
  
  // Draw circle around marker
  glColor4f(1.0f, 0.3f, 0.0f, alpha * 0.6f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 16; i++) {
    float angle = i * 2.0f * M_PI / 16.0f;
    glVertex2f(tx + (markerSize + 4.0f) * cosf(angle), 
               ty + (markerSize + 4.0f) * sinf(angle));
  }
  glEnd();
  
  glLineWidth(1.0f);
  glDisable(GL_BLEND);
}

inline void drawMinimap() {
  if (!showMinimap)
    return;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, windowWidth, 0, windowHeight);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);

  glColor4f(0.0f, 0.0f, 0.0f, 0.7f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBegin(GL_QUADS);
  glVertex2f(minimapX, minimapY);
  glVertex2f(minimapX + minimapSize, minimapY);
  glVertex2f(minimapX + minimapSize, minimapY + minimapSize);
  glVertex2f(minimapX, minimapY + minimapSize);
  glEnd();

  glColor3f(0.5f, 0.5f, 0.5f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(minimapX, minimapY);
  glVertex2f(minimapX + minimapSize, minimapY);
  glVertex2f(minimapX + minimapSize, minimapY + minimapSize);
  glVertex2f(minimapX, minimapY + minimapSize);
  glEnd();

  float scaleX = minimapSize / (2.0f * WORLD_SIZE);
  float scaleZ = minimapSize / (2.0f * WORLD_SIZE);
  float centerX = minimapX + minimapSize / 2.0f;
  float centerY = minimapY + minimapSize / 2.0f;

  glColor3f(1.0f, 0.0f, 0.0f);
  float ex = centerX + ENEMY_STATION.x * scaleX;
  float ey = centerY + ENEMY_STATION.z * scaleZ;
  glPointSize(8.0f);
  glBegin(GL_POINTS);
  glVertex2f(ex, ey);
  glEnd();

  glColor3f(0.0f, 1.0f, 0.0f);
  float dx = centerX + DEFENSE_STATION.x * scaleX;
  float dy = centerY + DEFENSE_STATION.z * scaleZ;
  glBegin(GL_POINTS);
  glVertex2f(dx, dy);
  glEnd();

  glColor4f(0.0f, 1.0f, 0.0f, 0.2f);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 32; i++) {
    float angle = i * 2.0f * M_PI / 32.0f;
    float rx = dx + RADAR_RANGE * scaleX * cosf(angle);
    float ry = dy + RADAR_RANGE * scaleZ * sinf(angle);
    glVertex2f(rx, ry);
  }
  glEnd();

  glPointSize(3.0f);
  glBegin(GL_POINTS);
  for (int i = 0; i < totalMissileCount; i++) {
    if (!h_missiles[i].active)
      continue;

    float mx = centerX + h_missiles[i].position.x * scaleX;
    float my = centerY + h_missiles[i].position.z * scaleZ;

    if (h_missiles[i].type == ENEMY_MISSILE) {
      glColor3f(1.0f, 0.0f, 0.0f);
    } else {
      glColor3f(0.0f, 0.8f, 1.0f);
    }
    glVertex2f(mx, my);
  }
  glEnd();

  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

inline void drawHUD() {
  CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, sizeof(TrainingMetrics),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_agent, d_agent, sizeof(RLAgent), cudaMemcpyDeviceToHost));

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0, windowWidth, 0, windowHeight);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);

  glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBegin(GL_QUADS);
  glVertex2f(windowWidth - 380, windowHeight - 320);
  glVertex2f(windowWidth - 10, windowHeight - 320);
  glVertex2f(windowWidth - 10, windowHeight - 10);
  glVertex2f(windowWidth - 380, windowHeight - 10);
  glEnd();
  glDisable(GL_BLEND);

  char buffer[256];
  int yPos = windowHeight - 35;

  glColor3f(0.0f, 1.0f, 1.0f);
  sprintf(buffer, "=== AI DEFENSE SYSTEM ===");
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 25;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(1.0f, 1.0f, 0.0f);
  sprintf(buffer, "Model: %s", modelName);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);

  glColor3f(1.0f, 1.0f, 1.0f);
  sprintf(buffer, "Mode: %s", trainingMode ? "TRAINING" : "OPERATIONAL");
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  sprintf(buffer, "Speed: %.0fx", simulationSpeed);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(1.0f, 0.3f, 0.3f);
  sprintf(buffer, "Enemy Missiles: %d", enemyMissileCount);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(0.3f, 0.8f, 1.0f);
  sprintf(buffer, "Interceptors: %d", interceptorCount);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 25;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(0.0f, 1.0f, 0.0f);
  sprintf(buffer, "Intercepts: %d", h_metrics->interceptSuccess);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(1.0f, 0.0f, 0.0f);
  sprintf(buffer, "Failures: %d", h_metrics->interceptFail);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  float successRate =
      (h_metrics->interceptSuccess + h_metrics->interceptFail > 0)
          ? (100.0f * h_metrics->interceptSuccess) /
                (h_metrics->interceptSuccess + h_metrics->interceptFail)
          : 0.0f;

  glColor3f(0.0f, 1.0f, 1.0f);
  sprintf(buffer, "Success Rate: %.1f%%", successRate);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  sprintf(buffer, "Best Rate: %.1f%%", h_agent->bestSuccessRate * 100.0f);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 25;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  sprintf(buffer, "RL Episodes: %d", h_agent->episodeCount);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  sprintf(buffer, "Epsilon: %.4f", h_agent->epsilon);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  sprintf(buffer, "Learn Rate: %.6f", h_agent->learningRate);
  glRasterPos2f(windowWidth - 360, yPos);
  yPos -= 20;
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(1.0f, 1.0f, 0.0f);
  sprintf(buffer, "FPS: %.1f", fps);
  glRasterPos2f(windowWidth - 360, yPos);

  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glColor3f(0.7f, 0.7f, 0.7f);
  sprintf(buffer, "Minimap Click: Launch | T: Train | S: Save | R: Reset");
  glRasterPos2f(20, 20);
  for (char *c = buffer; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

inline void displayGraph() {
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, graphWidth, 0, graphHeight);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glColor3f(0.05f, 0.05f, 0.1f);
  glBegin(GL_QUADS);
  glVertex2f(0, 0);
  glVertex2f(graphWidth, 0);
  glVertex2f(graphWidth, graphHeight);
  glVertex2f(0, graphHeight);
  glEnd();

  if (h_history->count < 2) {
    glutSwapBuffers();
    return;
  }

  float margin = 50.0f;
  float graphW = graphWidth - 2 * margin;
  float graphH = (graphHeight - 4 * margin) / 3.0f;

  // Success rate graph
  glColor3f(0.3f, 0.3f, 0.3f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(margin, margin);
  glVertex2f(margin + graphW, margin);
  glVertex2f(margin + graphW, margin + graphH);
  glVertex2f(margin, margin + graphH);
  glEnd();

  glColor3f(0.0f, 1.0f, 0.0f);
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < h_history->count; i++) {
    float x = margin + (float)i / (h_history->count - 1) * graphW;
    float y = margin + h_history->successRates[i] * graphH;
    glVertex2f(x, y);
  }
  glEnd();

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(margin, margin + graphH + 10);
  const char *label1 = "Success Rate";
  for (const char *c = label1; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);

  // Response time graph
  float yOffset = margin * 2 + graphH;
  glColor3f(0.3f, 0.3f, 0.3f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(margin, yOffset);
  glVertex2f(margin + graphW, yOffset);
  glVertex2f(margin + graphW, yOffset + graphH);
  glVertex2f(margin, yOffset + graphH);
  glEnd();

  float maxResponse = 10.0f;
  glColor3f(1.0f, 1.0f, 0.0f);
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < h_history->count; i++) {
    float x = margin + (float)i / (h_history->count - 1) * graphW;
    float y = yOffset +
              (1.0f - h_history->avgResponseTimes[i] / maxResponse) * graphH;
    glVertex2f(x, fmaxf(yOffset, y));
  }
  glEnd();

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(margin, yOffset + graphH + 10);
  const char *label2 = "Avg Response Time (lower is better)";
  for (const char *c = label2; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);

  // Reward graph
  yOffset = margin * 3 + graphH * 2;
  glColor3f(0.3f, 0.3f, 0.3f);
  glBegin(GL_LINE_LOOP);
  glVertex2f(margin, yOffset);
  glVertex2f(margin + graphW, yOffset);
  glVertex2f(margin + graphW, yOffset + graphH);
  glVertex2f(margin, yOffset + graphH);
  glEnd();

  float maxReward = 100.0f;
  float minReward = 0.0f;
  glColor3f(0.0f, 0.8f, 1.0f);
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i < h_history->count; i++) {
    float x = margin + (float)i / (h_history->count - 1) * graphW;
    float normalized =
        (h_history->rewards[i] - minReward) / (maxReward - minReward);
    float y = yOffset + normalized * graphH;
    glVertex2f(x, y);
  }
  glEnd();

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(margin, yOffset + graphH + 10);
  const char *label3 = "Episode Reward";
  for (const char *c = label3; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *c);

  // Title
  glColor3f(0.0f, 1.0f, 1.0f);
  glRasterPos2f(graphWidth / 2 - 80, graphHeight - 20);
  const char *title = "Training Metrics - Real-time";
  for (const char *c = title; *c; c++)
    glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);

  glutSwapBuffers();
}

#endif // RENDERER_H
