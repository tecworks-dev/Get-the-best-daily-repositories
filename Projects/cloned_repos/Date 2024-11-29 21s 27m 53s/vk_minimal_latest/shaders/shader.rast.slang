//------------------------------------------------------------------------------
// This shader demonstrates key Slang features and syntax while implementing
// a textured triangle renderer with dynamic point overlay effects
//------------------------------------------------------------------------------

#include "shader_io.h"

//------------------------------------------------------------------------------
// Input Structures
//------------------------------------------------------------------------------

// Define the vertex attributes that will be provided by the vertex buffer
// Note: In Slang, we use semantic annotations (: POSITION, etc.) to specify
// how these fields map to vertex buffer data
struct VertexInput
{
    [[vk::location(LVPosition)]] // Vulkan layout(location = 0)
    float3 position : POSITION;  // Vertex position in model space
    [[vk::location(LVColor)]]    // Vulkan layout(location = 1)
    float3 color : COLOR;        // Vertex color
    [[vk::location(LVTexCoord)]] // Vulkan layout(location = 2)
    float2 uv : TEXCOORD0;       // Texture coordinates
};

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
    float4 position : SV_Position; // Clip space position (required)
    float3 color : TEXCOORD0;      // Interpolated vertex color
    float2 uv : TEXCOORD1;         // Interpolated texture coordinates
};

// Define the final output of the fragment shader
// Note: SV_TARGET specifies this is the color output to the render target
struct PixelOutput
{
    float4 color : SV_TARGET; // Final color output (RGBA)
};

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Bind an array of textures
// Note: [[vk::binding(x,y)]] specifies binding=x in descriptor set=y for Vulkan
[[vk::binding(LBindTextures, LSetTextures)]]
Sampler2D inTexture[]; // Array of textures in descriptor set 0, binding 0

// Bind our scene information constant buffer
// Note: ConstantBuffer is similar to Vulkan's UBO (Uniform Buffer Object)
[[vk::binding(LBindSceneInfo, LSetScene)]]
ConstantBuffer<SceneInfo> sceneInfo; // Scene data in descriptor set 0, binding 1

// Bind push constants for frequently updated data
// Note: Push constants are fastest way to send small amount of data to shader
[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

// Specialization constant - can be set at pipeline creation time
// Note: This is equivalent to GLSL's layout(constant_id = X)
[[vk::constant_id(0)]]
const bool useTexture = false; // Controls whether texture sampling is enabled

//------------------------------------------------------------------------------
// Vertex Shader
//------------------------------------------------------------------------------

// Note: [shader("vertex")] is Slang's way of marking the vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    // Start with the input position
    vec3 pos = input.position;

    // Correct for screen aspect ratio to prevent stretching
    float aspectRatio = sceneInfo.resolution.y / sceneInfo.resolution.x;
    pos.x *= aspectRatio;

    VertexOutput output;
    output.position = float4(pos, 1.0); // Convert to homogeneous coordinates
    output.color    = input.color;      // Pass through the vertex color
    output.uv       = input.uv;         // Pass through the texture coordinates

    return output;
}

//------------------------------------------------------------------------------
// Fragment/Pixel Shader
//------------------------------------------------------------------------------

// Note: [shader("pixel")] is Slang's way of marking the fragment shader entry point
[shader("pixel")]
PixelOutput fragmentMain(VertexOutput input)
{
    PixelOutput output;

    // Convert fragment position to normalized device coordinates (-1 to 1 range)
    float2 fragPos = (input.position.xy / sceneInfo.resolution) * 2.0 - 1.0;

    // Access the data buffer using Vulkan's buffer device address feature
    // Note: This is a more direct way to access buffer data than traditional descriptors
    vec2* datas = (vec2*)sceneInfo.dataBufferAddress;

    // Find the minimum distance to any point in our data buffer
    float minDist = 1e10; // Initialize to a large value
    for (int i = 0; i < sceneInfo.numData; i++)
    {
        vec2  pnt  = datas[i];
        float dist = distance(fragPos, pnt);
        minDist    = min(minDist, dist);
    }

    // Create smooth circular points using distance field
    float radius     = 0.02; // Size of the points
    float edgeSmooth = 0.01; // Amount of anti-aliasing smoothing
    // Create smooth transition from 0 to 1 based on distance
    float alpha = 1.0 - smoothstep(radius, radius - edgeSmooth, minDist);

    // Calculate colors for blending
    // Point color animated using push constant data
    float4 pointColor = float4(sceneInfo.animValue * pushConst.color, 1.0);
    // Triangle color from interpolated vertex color
    float4 triangleColor = float4(input.color, 1.0);

    // Conditionally sample texture based on specialization constant
    if (useTexture)
    {
        triangleColor *= inTexture[sceneInfo.texId].Sample(input.uv);
    }

    // Blend between point and triangle color based on distance field
    output.color = lerp(pointColor, triangleColor, alpha);

    return output;
}
