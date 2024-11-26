cmake_minimum_required(VERSION 3.13)

project(NauAnalytics)

set(Sources
    nau_analytics.cpp nau_analytics.hpp
)

set(ProviderSources
    providers/nau_analytics_provider_interface.hpp
    providers/nau_analytics_provider_countly.hpp providers/nau_analytics_provider_countly.cpp
)

source_group("Sources" FILES ${Sources})
source_group("Sources\\providers" FILES ${ProviderSources})

add_library(
    NauAnalytics STATIC
    ${Sources}
    ${ProviderSources}
)

target_include_directories(NauAnalytics PRIVATE
    "${CMAKE_SOURCE_DIR}/editor/external/countly/include"
    "${CMAKE_SOURCE_DIR}/editor/external/countly/vendor/json/include"
    "${CMAKE_SOURCE_DIR}/editor/external/libzip/lib"
    "${CMAKE_BINARY_DIR}/editor/external/libzip"
    "${CMAKE_SOURCE_DIR}/editor/external/libcurl/include/"
)

target_link_libraries(NauAnalytics PRIVATE countly crypto zlibstatic zip libcurl)
target_compile_definitions(NauAnalytics PRIVATE COUNTLY_USE_SQLITE)

set_target_properties(NauAnalytics PROPERTIES
    # Countly requires this option to be disabled
    COMPILE_WARNING_AS_ERROR OFF
)

set_target_properties(NauAnalytics PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/NauAnalytics)
set_target_properties(NauAnalytics PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/NauAnalytics)
