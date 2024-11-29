#version 460
#extension GL_EXT_buffer_reference2 : require                 // buffer reference
#extension GL_EXT_scalar_block_layout : require               // scalar layout
#extension GL_GOOGLE_include_directive : require              // #include
#extension GL_EXT_shader_explicit_arithmetic_types : require  // uint64_t

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#include "shader_io.h"


// Push constant containing the rotation angle and the address of the buffer
layout(push_constant, scalar) uniform PushConstant_
{
  PushConstantCompute pushConst;
};

// Buffer reference, address of buffer in the push constant
layout(buffer_reference, scalar) readonly buffer Vertex_
{
  Vertex _[];
};


void main()
{
  // Get the index of the current work item
  uint index = gl_GlobalInvocationID.x;

  // Early exit
  if(index >= pushConst.numVertex)
    return;

  Vertex_ vertices = Vertex_(pushConst.bufferAddress);

  // Each vertex will rotate at a different speed
  float angle = pushConst.rotationAngle * sin((index + 1) * 2.3f);

  // Retrieve the vertex at the current index
  vec3 vertex = vertices._[index].position;

  // Compute the rotation matrix around the Z-axis
  float cosAngle       = cos(angle);
  float sinAngle       = sin(angle);
  mat3  rotationMatrix = mat3(cosAngle, -sinAngle, 0.0, sinAngle, cosAngle, 0.0, 0.0, 0.0, 1.0);

  // Apply the rotation to the vertex
  vec3 rotatedVertex = rotationMatrix * vertex;

  // Store the rotated vertex back in the buffer
  vertices._[index].position = rotatedVertex;
}
