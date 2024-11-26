add_subdirectory(source/editor/ui)
add_subdirectory(source/editor/unit)

set_target_properties (
  NauEditorUiTests NauEditorUnitTests
  PROPERTIES
    FOLDER "${NauEditorFolder}/tests"
)