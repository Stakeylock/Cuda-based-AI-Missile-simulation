#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "../core/config.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Device math functions
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a) {
  return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator/(const float3 &a, float s) {
  return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3 &v) { return sqrtf(dot(v, v)); }

__host__ __device__ inline float3 normalize(const float3 &v) {
  float len = length(v);
  return len > 1e-6f ? v / len : make_float3(0.0f, 0.0f, 0.0f);
}

// Cross product for 3D vectors
__host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

// Activation functions
__device__ inline float relu(float x) { return fmaxf(0.0f, x); }

__device__ inline float tanh_activation(float x) { return tanhf(x); }

#endif // CUDA_UTILS_CUH
