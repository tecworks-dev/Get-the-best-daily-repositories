
SET(ADS_VERSION "0.0.1" CACHE BOOL "" FORCE)
SET(ENABLE_BZIP2 OFF  CACHE BOOL "" FORCE)
SET(ENABLE_BZIP2 OFF  CACHE BOOL "" FORCE)
SET(ENABLE_LZMA OFF CACHE BOOL "" FORCE)
SET(ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
SET(ENABLE_GNUTLS OFF CACHE BOOL "" FORCE)
SET(ENABLE_MBEDTLS OFF CACHE BOOL "" FORCE)
SET(BUILD_DOC OFF CACHE BOOL "" FORCE)
SET(BUILD_REGRESS OFF CACHE BOOL "" FORCE)
SET(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
SET(BUILD_TESTING OFF CACHE BOOL "" FORCE)
SET(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
SET(BUILD_STATIC OFF CACHE BOOL "" FORCE)
SET(ZIP_STATIC ON CACHE BOOL "" FORCE)
SET(CURL_USE_SSL ON CACHE BOOL "" FORCE)
SET(CURL_STATICLIB ON CACHE BOOL "" FORCE)
SET(CURL_ENABLE_SSL ON CACHE BOOL "" FORCE)
SET(CURL_USE_OPENSSL OFF CACHE BOOL "" FORCE)
SET(CURL_ENABLE_EXPORT_TARGET OFF CACHE BOOL "" FORCE)
SET(LIBZIP_DO_INSTALL OFF CACHE BOOL "" FORCE)
SET(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
SET(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(COUNTLY_USE_SQLITE ON CACHE BOOL "" FORCE)

# Note ADS should be integrated as shared.
# Static linkage approach does not fit very well with the engine's memory management subsystem.
add_subdirectory(ads EXCLUDE_FROM_ALL)
set_target_properties(advanceddocking PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/NauEditor)
set_target_properties(advanceddocking PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/NauEditor)

file(COPY "${CMAKE_SOURCE_DIR}/editor/external/opensslconfig/apps" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/openssl")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/opensslconfig/crypto" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/openssl")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/opensslconfig/include" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/openssl")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/opensslconfig/providers" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/openssl")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/opensslconfig/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/openssl")

file(COPY "${CMAKE_SOURCE_DIR}/editor/external/libcurlconfig/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/libcurl")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/libzipconfig/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/libzip")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/zlibconfig/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/zlib")

file(COPY "${CMAKE_SOURCE_DIR}/editor/external/countlyconfig/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/countly")
file(COPY "${CMAKE_SOURCE_DIR}/editor/external/countlyconfig/vendor/sqlite/CMakeLists.txt" DESTINATION "${CMAKE_SOURCE_DIR}/editor/external/countly/vendor/sqlite")

add_subdirectory(zlib EXCLUDE_FROM_ALL)

set(ZLIB_INCLUDE_DIR  "${CMAKE_CURRENT_BINARY_DIR}/zlib" "${CMAKE_CURRENT_LIST_DIR}/zlib")
set(ZLIB_INCLUDE_DIR  "${ZLIB_INCLUDE_DIR}" PARENT_SCOPE)
set(ZLIB_LIBRARY_DEBUG   "${CMAKE_CURRENT_BINARY_DIR}/zlib/Debug/zlibstaticd.lib")
set(ZLIB_LIBRARY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/zlib/Release/zlibstatic.lib")
set(ZLIB_LIBRARY_RELEASE "${ZLIB_LIBRARY_RELEASE}"  PARENT_SCOPE)
set(ZLIB_LIBRARY_DEBUG   "${ZLIB_LIBRARY_DEBUG}"  PARENT_SCOPE)

set(LIBZIP_INCLUDE_DIR   "${CMAKE_CURRENT_LIST_DIR}/libzip/lib")

add_subdirectory(libzip EXCLUDE_FROM_ALL)
add_subdirectory(magic_enum EXCLUDE_FROM_ALL)
add_subdirectory(openssl EXCLUDE_FROM_ALL)

set(OPENSSL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/openssl/include")
set(OPENSSL_INCLUDE_DIR "${OPENSSL_INCLUDE_DIR}" PARENT_SCOPE)

set(OPENSSL_LIBRARIES   "${CMAKE_CURRENT_BINARY_DIR}/openssl/$<CONFIG>/openssl.lib" )
set(OPENSSL_LIBRARIES   "${OPENSSL_LIBRARIES}"  PARENT_SCOPE)

add_subdirectory(libcurl EXCLUDE_FROM_ALL)

target_include_directories(libcurl PRIVATE "${CMAKE_CURRENT_LIST_DIR}/openssl/include")
target_link_libraries(libcurl PRIVATE openssl)
target_compile_definitions(libcurl  PRIVATE USE_OPENSSL)


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/countly/vendor/sqlite EXCLUDE_FROM_ALL)
set(SQLite3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/countly/vendor/sqlite)
set(SQLite3_LIBRARY sqlite3)

add_subdirectory(countly EXCLUDE_FROM_ALL)

set(COUNTLY_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/countly/include")
set(COUNTLY_INCLUDE_DIR "${COUNTLY_INCLUDE_DIR}" PARENT_SCOPE)

set(COUNTLY_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/countly/$<CONFIG>/countly.lib")
set(COUNTLY_LIBRARIES "${COUNTLY_LIBRARIES}" PARENT_SCOPE)

# Define the COUNTLY_USE_SQLITE macro for the countly target
target_compile_definitions(countly PRIVATE COUNTLY_USE_SQLITE)

foreach(target advanceddocking zlib dist distcheck minigzip
        zip  zlibstatic  example crypto curl_uninstall libcurl openssl countly sqlite3)
    set_target_properties("${target}" PROPERTIES FOLDER "${NauEditorFolder}/external")
endforeach()