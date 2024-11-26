
# Temporary. We have the engine now only in debug.
OPTION(NAU_ENGINE_FROM_SOURCE "if OFF then use prebuild engine path form env variable NAU_ENGINE_DIR" ON)

IF (NOT DEFINED NAU_ENGINE_DIR)
    SET(NAU_ENGINE_DIR ${CMAKE_SOURCE_DIR}/engine)
ENDIF()

nau_collect_files(Sources
  DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/src
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  MASK "*.cpp" "*.hpp"
)

nau_collect_files(PublicHeaders
  DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/nau"
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  MASK "*.hpp"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Sources" FILES ${Sources})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Headers" FILES ${PublicHeaders})

add_library(
    NauEditorEngineLog SHARED
    ${Sources}
    ${PublicHeaders}
)

target_include_directories(NauEditorEngineLog PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

target_compile_definitions(NauEditorEngineLog PRIVATE
    NAU_EDITOR_ENGINE_LOG_BUILD_DLL
)

target_link_options(NauEditorEngineLog
    PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/FORCE:MULTIPLE>

    # Note that since MSVC 2019, /DEBUG:FULL linking times have improved significantly,
    # /DEBUG:FASTLINK isn't always faster than /DEBUG:FULL.
    # We'd rather have more clear debug picture than a small chance of faster building.
    PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/DEBUG:FULL>

    # OPT:REF can reduce the size of executable, for it eliminates functions and data that are never referenced.
    PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:REF>

    # OPT:NOICF  makes it easier to read stack traces and set breakpoints in functions.
    PRIVATE $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:ICF>
)

if(MSVC)
    target_compile_options(NauEditorEngineLog PRIVATE "-wd4595" "$<$<CONFIG:Release>:/Zi>")
endif()

set_target_properties(NauEditorEngineLog PROPERTIES AUTOMOC TRUE)

target_link_libraries(NauEditorEngineLog PRIVATE
    Qt6::Widgets

    NauEditorEngine
    NauFramework
    PlatformApp
)

nau_add_compile_options(NauEditorEngineLog ENABLE_RTTI)

set_target_properties(NauEditorEngineLog PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/NauEditor)
set_target_properties(NauEditorEngineLog PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/NauEditor)

if (TARGET NauKernel_Import AND TARGET NauFramework_Import)
  add_dependencies(${TargetName} NauKernel_Import NauFramework_Import)
endif()
