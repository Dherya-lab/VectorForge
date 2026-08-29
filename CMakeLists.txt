cmake_minimum_required(VERSION 3.20)

project(VectorForge
    VERSION 0.1.0
    DESCRIPTION "High-performance C++20 vector search engine"
    LANGUAGES CXX
)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(core)
