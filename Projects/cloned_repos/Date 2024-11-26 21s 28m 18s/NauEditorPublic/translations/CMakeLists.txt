set(
    ts_files
    editor_en.ts
    editor_ru.ts
)
set(ProjectName NauEditor)

nau_collect_files(Sources
  DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../editor/source/
  MASK "*.cpp"
)

qt_add_translations(${ProjectName} TS_FILES ${ts_files} 
    SOURCES ${Sources}
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../editor/source/
    LUPDATE_OPTIONS -locations none -no-obsolete -recursive
)

add_dependencies(${ProjectName} ${ProjectName}_lupdate)

foreach(target ${ProjectName}_lupdate ${ProjectName}_lrelease ${ProjectName}_${ProjectName}_translations ${ProjectName}_other_files
    release_translations update_translations)
    set_target_properties("${target}" PROPERTIES FOLDER "i18n")
endforeach()