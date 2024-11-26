cmake_minimum_required(VERSION 3.13)

project(NauEditorUiTests)
enable_testing()

# Needed for making NauEditorUiTests project without NauEditor
IF (NOT Qt6_FOUND)
    find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network Test)
ENDIF()

IF (NOT DEFINED NAU_EDITOR_SOURCE_DIR)
    set(NAU_EDITOR_SOURCE_DIR "${CMAKE_SOURCE_DIR}/editor/source/editor-core")
ENDIF()

set(Sources
    nau_main.cpp
    nau_tests.cpp
    nau_test_flow_layout.cpp
)

set(EditorSources
    ${NAU_EDITOR_SOURCE_DIR}/nau_flow_layout.cpp ${NAU_EDITOR_SOURCE_DIR}/nau_flow_layout.hpp
)

set(Headers
    nau_tests.hpp
    nau_test_flow_layout.hpp
)

add_executable(NauEditorUiTests WIN32 ${Sources} ${EditorSources} ${Headers})

target_include_directories(NauEditorUiTests PRIVATE
 "${NAU_EDITOR_SOURCE_DIR}"
 "${NAU_EDITOR_SOURCE_DIR}/baseWidgets"
 "${NAU_EDITOR_SOURCE_DIR}/include"
)

target_compile_definitions(NauEditorUiTests PRIVATE
    NAU_UNIT_TESTS
    NAU_EDITOR_STATIC_RUNTIME
)

enable_testing()
target_link_libraries(NauEditorUiTests Qt6::Widgets Qt6::Test)
set_target_properties(NauEditorUiTests PROPERTIES AUTOMOC TRUE)
set_target_properties(NauEditorUiTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/NauEditorUiTests)
set_target_properties(NauEditorUiTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/NauEditorUiTests)
