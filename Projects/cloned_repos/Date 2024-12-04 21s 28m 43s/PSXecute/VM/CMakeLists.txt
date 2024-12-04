cmake_minimum_required(VERSION 3.25.1)
project(PSXecute)

if (LINUX)
    set (CMAKE_C_FLAGS "-m32")
    set (CMAKE_CXX_FLAGS "-m32 -Wno-shift-count-overflow")
    set(CMAKE_C_COMPILER i686-w64-mingw32-gcc)
    set(CMAKE_CXX_COMPILER i686-w64-mingw32-g++)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libstdc++ -static-libgcc")
endif()

# debug build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-D_DEBUG)
endif()

add_executable(PSXecute
    main.cpp
    cpu/Psx.cpp
    cpu/Psx.h
    cpu/Instruction.cpp
    cpu/Instruction.h
    cpu/Opcodes.cpp
)
