add_subdirectory(editor-engine)
add_subdirectory(editor-core)
add_subdirectory(editor-engine-log)
add_subdirectory(editor-core/analytics/source)
add_subdirectory(editor-plugin-manager)
add_subdirectory(editor)

set_target_properties (
  NauEditor NauEditorCore NauEditorEngine
  NauEditorEngineLog NauAnalytics
  NauEditorPluginManager
  PROPERTIES
    FOLDER "${NauEditorFolder}/source"
)

nau_collect_cmake_subdirectories(editorModules ${CMAKE_CURRENT_SOURCE_DIR}/modules)

foreach(module ${editorModules})
  message(STATUS "Editor module: (${module})")
  add_subdirectory(modules/${module})
endforeach()