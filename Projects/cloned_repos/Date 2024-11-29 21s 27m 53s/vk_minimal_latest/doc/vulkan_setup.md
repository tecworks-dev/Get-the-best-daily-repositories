# Vulkan Setup and Rendering Pipeline

This document provides a step-by-step guide to setting up a Vulkan context, creating a swapchain, setting up the graphics pipeline, creating a vertex buffer for geometry, and briefly covering shaders. This guide is intended for beginners in graphics programming and Vulkan usage.

## 1. Vulkan Context Creation

The first step in using Vulkan is to create a Vulkan context. This involves initializing the Vulkan library and setting up the necessary Vulkan objects.

### Steps:
1. **Create a Vulkan Instance**: This is the connection between your application and the Vulkan library.
1. **Select a Physical Device**: Choose a GPU that supports Vulkan.
1. **Create a Logical Device**: This represents the GPU and allows you to interact with it.

## 2. GLFW Initialization and Surface

With GLFW initialized and having a Vulkan context, the next step is to create a window surface to render graphics to.

1. **Create a Surface**: This is the window or display where the graphics will be rendered.


## 3. Swapchain Creation

The swapchain is a series of images that are presented to the screen in sequence. It is essential for rendering graphics smoothly.

### Steps:
1. **Query Swapchain Support**: Check the capabilities of the swapchain.
2. **Choose Swapchain Settings**: Select the surface format, present mode, and swap extent.
3. **Create the Swapchain**: Create the swapchain with the chosen settings.


## 4. Graphics Pipeline Creation

The graphics pipeline is a sequence of operations that process vertex data and produce rendered images.

### Steps:
1. **Create Shader Modules**: Load and compile shaders.
2. **Create Pipeline Layout**: Define the layout of the pipeline.
3. **Create Render Pass**: Define how rendering operations are performed.
4. **Create Graphics Pipeline**: Combine all the settings into a pipeline.


## 5. Vertex Buffer Creation

The vertex buffer holds the geometry data that will be rendered.

### Steps:
1. **Define Vertex Data**: Specify the vertices of the geometry.
2. **Create Vertex Buffer**: Allocate memory and copy the vertex data to the buffer.


## 6. Shaders

Shaders are compiled using glslangValidator to Spir-V in a header. This avoid to compile the shaders on the fly at runtime.

