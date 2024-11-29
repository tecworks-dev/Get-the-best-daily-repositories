//------------------------------------------------------------------------------
// Compute shader demonstrating vertex transformation and buffer device address usage
// This shader rotates vertices at different speeds based on their index
//------------------------------------------------------------------------------

#include "shader_io.h"

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Output texture that can be both read and written
// Note: RWTexture2D is equivalent to Vulkan's storage image
[[vk::binding(0)]] 
RWTexture2D<float4> outImage;

// Push constants for frequently updated data
// Note: Using push constants is more efficient than constant buffers for small, frequently changing data
[[vk::push_constant]]  
ConstantBuffer<PushConstantCompute> pushConst;

//------------------------------------------------------------------------------
// Compute Shader Entry Point
//------------------------------------------------------------------------------

// Declare this as a compute shader with [shader("compute")]
// numthreads specifies the compute shader thread group size:
// - 256 threads in X dimension
// - 1 thread in Y dimension
// - 1 thread in Z dimension
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)  // SV_DispatchThreadID gives us the global thread ID
{
    // Extract the x component which we use as our 1D index
    // Note: In a compute shader, each thread operates on one element
    uint index = threadIdx.x;
    
    // Exit if this thread is beyond our data range
    // Note: This is important as we might dispatch more threads than we have vertices
    if(index >= pushConst.numVertex)
        return;
    
    // Access the vertex buffer using buffer device address
    // Note: This is using Vulkan's VK_KHR_buffer_device_address extension
    // allowing direct pointer access to GPU buffers
    Vertex* vertices = (Vertex*)pushConst.bufferAddress;
    
    // Calculate a unique rotation angle for each vertex
    // - Base angle comes from pushConst.rotationAngle
    // - Multiplied by sin((index + 1) * 2.3f) to create varying speeds
    // - The 2.3f factor creates a non-uniform distribution of speeds
    float angle = pushConst.rotationAngle * sin((index + 1) * 2.3f);
    
    // Load the current vertex position from the buffer
    float3 vertex = vertices[index].position;
    
    // Construct a 3D rotation matrix around the Z-axis
    // | cos θ  -sin θ   0 |
    // | sin θ   cos θ   0 |
    // |   0      0      1 |
    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    float3x3 rotationMatrix = float3x3(
        cosAngle, -sinAngle, 0.0,   // First row
        sinAngle,  cosAngle, 0.0,   // Second row
        0.0,       0.0,      1.0    // Third row
    );
    
    // Apply the rotation matrix to the vertex position
    // Note: Using matrix multiplication to transform the position
    float3 rotatedVertex = mul(rotationMatrix, vertex);
    
    // Write the transformed position back to the buffer
    // Note: This will be visible to subsequent graphics operations
    vertices[index].position = rotatedVertex;
}