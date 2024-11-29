#version 460
#extension GL_GOOGLE_include_directive : require              // For #include
#extension GL_EXT_scalar_block_layout : require               // For scalar layout
#extension GL_EXT_shader_explicit_arithmetic_types : require  // For uint64_t, ...

#include "shader_io.h"


layout(location = LVPosition) in vec3 inPosition;
layout(location = LVColor) in vec3 inColor;
layout(location = LVTexCoord) in vec2 inUv;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outUv;

layout(set = 1, binding = 0, scalar) uniform _SceneInfo
{
  SceneInfo sceneInfo;
};


void main()
{
  vec3 pos = inPosition;

  // Adjust aspect ratio
  float aspectRatio = sceneInfo.resolution.y / sceneInfo.resolution.x;
  pos.x *= aspectRatio;
  // Set the position in clip space
  gl_Position = vec4(pos, 1.0);
  // Pass the color and uv
  outColor = inColor;
  outUv    = inUv;
}