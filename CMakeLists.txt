cmake_minimum_required(VERSION 3.14)
project(flashLLM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gen compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# third-party extentions
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
)
FetchContent_MakeAvailable(fmt)

file(GLOB_RECURSE SOURCE_FILE_LISTS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADER_FILE_LISTS ${PROJECT_SOURCE_DIR}/src/*.hpp)
# message(STATUS "sources: ${SOURCE_FILE_LISTS}")
# message(STATUS "headers: ${HEADER_FILE_LISTS}")

add_executable(flashLLM main.cpp ${SOURCE_FILE_LISTS} ${HEADER_FILE_LISTS})
target_link_libraries(flashLLM PRIVATE fmt::fmt)
