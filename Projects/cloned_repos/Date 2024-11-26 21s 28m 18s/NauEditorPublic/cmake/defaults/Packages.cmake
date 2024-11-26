# file to find packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network Test OpenGL LinguistTools Svg)

#Generating BUILD_INFO.txt 
find_package(Git)
if(Git_FOUND)

  #short hash for "Help/About" diolog window
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" "rev-parse" "--short" "HEAD"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NAUEDITOR_REPO_SHORT_HASH
  )
  add_compile_definitions(NAU_COMMIT_HASH="${NAUEDITOR_REPO_SHORT_HASH}")
  
else()
  message(WARNING "Git NOT found! BUILD_INFO.txt won't be created")
endif()