cmake_minimum_required(VERSION 3.13)

if(POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif() 

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD ON)
set(QT_VERSION 6)
set(CMAKE_CXX_STANDARD 20)

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

project(NauEditor)

if(NOT DEFINED ${NauEditorFolder})
  set(NauEditorFolder "NauEditor")
endif(NOT DEFINED ${NauEditorFolder})

# In multi-configuration IDEs, only allow Debug and Release
if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES Debug Release)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Build Configurations" FORCE)
endif()

if(NOT DEFINED NAU_ROOT_DIR)
  set(NAU_ROOT_DIR "$ENV{NAU_ENGINE_SOURCE_DIR}")
endif()

cmake_path(SET NAU_ROOT_DIR NORMALIZE "${NAU_ROOT_DIR}")

set(NAU_CORE_TESTS OFF)
# Shaders are assembled in samples, so we temporarily include them in the assembly.
# TODO: Remove them from the assembly when shader generation is transferred to engine modules
set(NAU_CORE_SAMPLES ON)
set(NAU_CORE_TOOLS ON)

set(CMAKE_MODULE_PATH
  "${NAU_ROOT_DIR}/cmake"
  "${CMAKE_SOURCE_DIR}/cmake/defaults"
  "${CMAKE_SOURCE_DIR}/cmake/macros"
)

add_subdirectory("${NAU_ROOT_DIR}" "NauEngine")

include(defaults/UsdPackages)
include(NauCommon)
include(NauModule)
include(NauGenFunctions)
include(platformSpec/NauPlatformSetup)
include(Packages)

add_subdirectory(editor)
add_subdirectory(installer)
add_subdirectory(translations)
