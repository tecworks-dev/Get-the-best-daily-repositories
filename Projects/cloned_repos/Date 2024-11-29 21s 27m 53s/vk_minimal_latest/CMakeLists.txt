cmake_minimum_required(VERSION 3.21)
set(MYPROJ vk_minimal_latest)
project(${MYPROJ} VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Download CPM.cmake
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Find Vulkan
find_package(Vulkan REQUIRED)

if((NOT Vulkan_GLSLANG_VALIDATOR_EXECUTABLE) OR (NOT Vulkan_GLSLC_EXECUTABLE))
    get_filename_component(_Vulkan_LIB_DIR ${Vulkan_LIBRARY} DIRECTORY)
    find_program(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE
        NAMES glslangValidator
        HINTS ${_Vulkan_LIB_DIR}/../Bin)
    find_program(Vulkan_GLSLC_EXECUTABLE
        NAMES glslc
        HINTS ${_Vulkan_LIB_DIR}/../Bin)
endif()

# Add the executable
add_executable(${MYPROJ} 
    src/minimal_latest.cpp 
    src/debug_util.h
    src/logger.h
    )

# Make Visual Studio using this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${MYPROJ})

# Set warning level 3 for MSVC
if(MSVC)
  target_compile_options(${MYPROJ} PRIVATE /W4)
endif()

# Include GLFW
CPMAddPackage(
  NAME glfw
  GITHUB_REPOSITORY glfw/glfw
  GIT_TAG 3.4
  OPTIONS
    "GLFW_BUILD_DOCS OFF"
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_WAYLAND OFF"
    "GLFW_INSTALL OFF"
)

# Include GLM
CPMAddPackage(
  NAME glm
  GITHUB_REPOSITORY g-truc/glm
  GIT_TAG 1.0.1
)

# Include Volk
CPMAddPackage(
  NAME volk
  GITHUB_REPOSITORY zeux/volk
  GIT_TAG 1.3.295
)

# Include VMA
CPMAddPackage(
  NAME VulkanMemoryAllocator
  GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  GIT_TAG v3.1.0
)

#include ImGui
CPMAddPackage(
  NAME imgui
  GITHUB_REPOSITORY ocornut/imgui
  GIT_TAG docking
)
set(IMGUI_SOURCES 
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui.h
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.h
)
target_include_directories(${MYPROJ} PRIVATE ${imgui_SOURCE_DIR})
target_sources(${MYPROJ} PRIVATE ${IMGUI_SOURCES})
target_compile_definitions(${MYPROJ} PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)
source_group("Imgui" FILES ${IMGUI_SOURCES})

# Download stb_image.h
set(STB_IMAGE_URL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h")
set(STB_IMAGE_PATH "${CMAKE_BINARY_DIR}/stb_image.h")
if(NOT EXISTS ${STB_IMAGE_PATH})
    message(STATUS "Downloading stb_image.h...")
    file(DOWNLOAD ${STB_IMAGE_URL} ${STB_IMAGE_PATH})
endif()
target_include_directories(${MYPROJ} PRIVATE "${CMAKE_BINARY_DIR}")

# Link libraries and include directories
target_link_libraries(${MYPROJ} PRIVATE
  glfw
  volk
  VulkanMemoryAllocator
)

target_include_directories(${MYPROJ} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${glm_SOURCE_DIR}
  ${imgui_SOURCE_DIR}
  ${VulkanMemoryAllocator_SOURCE_DIR}/include
)

# Define VK_NO_PROTOTYPES to avoid including Vulkan prototypes
# This is necessary because we are using volk to load Vulkan functions
target_compile_definitions(${MYPROJ} PRIVATE VK_NO_PROTOTYPES)


#------------------------------------------------------------------------------------------------------------------------------
# Compile GLSL shaders
file(GLOB SHADER_FILES "shaders/*.glsl")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/_autogen)
foreach(SHADER ${SHADER_FILES})
  get_filename_component(SHADER_NAME ${SHADER} NAME)
  string(REPLACE "." "_" VN_SHADER_NAME ${SHADER_NAME})
  set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/_autogen/${SHADER_NAME}.h")
  set(_COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -g -D_glsl --target-env vulkan1.3 --vn ${VN_SHADER_NAME} -o ${OUTPUT_FILE} ${SHADER})
  add_custom_command(
    PRE_BUILD
    OUTPUT ${OUTPUT_FILE}
    COMMAND ${_COMMAND}
    # COMMENT  "${_COMMAND}"
    MAIN_DEPENDENCY ${SHADER}
    COMMENT "Compiling GLSL shader ${SHADER_NAME}"
    VERBATIM
  )
  list(APPEND SHADER_HEADERS ${OUTPUT_FILE})
endforeach()

#------------------------------------------------------------------------------------------------------------------------------
# Find Slang compiler
find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS ${_Vulkan_LIB_DIR}/../Bin
)
if( SLANGC_EXECUTABLE)
    # Initialize variables
    set(USE_GLSL OFF )
    set(USE_SLANG OFF)
    # Define the available choices for shader languages
    set(CHOICES "GLSL" "SLANG")
    set(USE_SHADER_LANGUAGE "SLANG" CACHE STRING "Choose the shading language to use.") # Shader language to use by default
    set_property(CACHE USE_SHADER_LANGUAGE PROPERTY STRINGS ${CHOICES}) # Provide the choices to the user in the CMake GUI
    if(USE_SHADER_LANGUAGE STREQUAL "SLANG")
        target_compile_definitions(${MYPROJ} PRIVATE USE_SLANG)
    endif()


    set(_SLANG_FLAGS
        -profile sm_6_6+spirv_1_6
        -capability spvInt64Atomics+spvShaderInvocationReorderNV+spvShaderClockKHR+spvRayTracingMotionBlurNV
        -target spirv
        -emit-spirv-directly
        -force-glsl-scalar-layout
        -fvk-use-entrypoint-name
        -g3
        -source-embed-style text     
    )

    # Compile Slang shaders using slangc
    file(GLOB SHADER_SLANG_FILES "shaders/*.slang")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/_autogen)
    foreach(SHADER ${SHADER_SLANG_FILES})
      get_filename_component(SHADER_NAME ${SHADER} NAME)
      string(REPLACE "." "_" VN_SHADER_NAME ${SHADER_NAME})
      set(OUTPUT_FILE "${CMAKE_BINARY_DIR}/_autogen/${SHADER_NAME}.h")
      set(_COMMAND ${SLANGC_EXECUTABLE} 
        ${_SLANG_FLAGS} 
        -source-embed-name ${VN_SHADER_NAME}
        -o ${OUTPUT_FILE} ${SHADER})
      add_custom_command(
        PRE_BUILD
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${_COMMAND}
        MAIN_DEPENDENCY ${SHADER}
        COMMENT "Compiling Slang shader ${SHADER_NAME} with slangc"
        VERBATIM
      )
      list(APPEND SHADER_HEADERS ${OUTPUT_FILE})
    endforeach()


else()
    message(ERROR "slangc executable not found")
endif()



# Add shader files to the project
file(GLOB SHADER_H_FILES "shaders/*.h")
source_group("Shaders" FILES ${SHADER_FILES} ${SHADER_SLANG_FILES} ${SHADER_H_FILES})
target_sources(${MYPROJ} PRIVATE ${SHADER_FILES} ${SHADER_H_FILES})

# Add the shader headers directly to the executable target
target_sources(${MYPROJ} PRIVATE ${SHADER_HEADERS})



# Include the _autogen directory
target_include_directories(${MYPROJ} PRIVATE "${CMAKE_BINARY_DIR}")



# Copy .jpg image next to the executable
file(GLOB IMAGE_FILE "${CMAKE_SOURCE_DIR}/resources/*.jpg")
add_custom_command(
  TARGET ${MYPROJ} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${IMAGE_FILE} $<TARGET_FILE_DIR:${MYPROJ}>
)

# Set a default installation prefix
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_install" CACHE PATH "Default install path" FORCE)

# Define the installation directory for the executable
install(TARGETS ${MYPROJ}
    RUNTIME DESTINATION bin
)

# Installing also the image
install(FILES ${IMAGE_FILE}
    DESTINATION bin
)
