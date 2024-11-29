
#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __SLANG__
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
#define STATIC_CONST static const
#else
#define STATIC_CONST const
#endif

// Layout constants
// Set 0
STATIC_CONST int LSetTextures  = 0;
STATIC_CONST int LBindTextures = 0;
// Set 1
STATIC_CONST int LSetScene      = 1;
STATIC_CONST int LBindSceneInfo = 0;
// Vertex layout
STATIC_CONST int LVPosition = 0;
STATIC_CONST int LVColor    = 1;
STATIC_CONST int LVTexCoord = 2;


struct SceneInfo
{
  uint64_t dataBufferAddress;
  vec2     resolution;
  float    animValue;
  int      numData;
  int      texId;
};

struct PushConstant
{
  vec3 color;
};

struct PushConstantCompute
{
  uint64_t bufferAddress;
  float    rotationAngle;
  int      numVertex;
};

struct Vertex
{
  vec3 position;
  vec3 color;
  vec2 texCoord;
};


#endif  // HOST_DEVICE_H
