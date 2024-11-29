#version 460

#extension GL_GOOGLE_include_directive : require              // For #include
#extension GL_EXT_scalar_block_layout : require               // For scalar layout
#extension GL_EXT_shader_explicit_arithmetic_types : require  // For uint64_t, ...
#extension GL_EXT_buffer_reference2 : require                 // For buffer reference
#extension GL_EXT_nonuniform_qualifier : require              // For non-uniform indexing of the texture array

#include "shader_io.h"

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

// From the descriptor set
layout(set = LSetTextures, binding = LBindTextures) uniform sampler2D inTexture[];
layout(set = LSetScene, binding = LBindSceneInfo, scalar) uniform _SceneInfo
{
  SceneInfo sceneInfo;
};

// Defined when creating the pipeline
layout(push_constant, scalar) uniform PushConstant_
{
  PushConstant pushConst;
};

// Buffer reference, address of buffer in the push constant
layout(buffer_reference, scalar) readonly buffer Datas
{
  vec2 _[];
};

// Specialization constant
layout(constant_id = 0) const bool useTexture = false;


void main()
{
  // Compute the normalized fragment position and center it at (0, 0)
  vec2 fragPos = (gl_FragCoord.xy / sceneInfo.resolution) * 2.0 - 1.0;

  // Access the data buffer
  Datas datas = Datas(sceneInfo.dataBufferAddress);

  // Loop over points in the data buffer
  // Compute the distance between the fragment and each point (uniform screen space, not moving with triangle)
  float minDist = 1e10;
  for(int i = 0; i < sceneInfo.numData; i++)
  {
    vec2  pnt  = datas._[i];
    float dist = distance(fragPos, pnt);
    minDist    = min(minDist, dist);
  }

  // Create a smooth transition around the points' boundaries (anti-aliasing effect)
  float radius     = 0.02;
  float edgeSmooth = 0.01;  // Smooth the edge
  float alpha      = 1.0 - smoothstep(radius, radius - edgeSmooth, minDist);

  vec4 pointColor = vec4(sceneInfo.animValue * pushConst.color, 1.0);  // points flashing using the value from the push constant
  vec4 triangleColor = vec4(fragColor, 1.0);                           // Interpolated color from the vertex shader

  if(useTexture)
    triangleColor *= texture(inTexture[sceneInfo.texId], inUv);

  // Blend the point with the background based on the minimum distance
  outColor = mix(pointColor, triangleColor, alpha);
}
