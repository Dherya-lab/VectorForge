add_library(vectorforge_core
    src/vector.cpp
)

target_include_directories(vectorforge_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vectorforge_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_core PRIVATE /W4)
else()
    target_compile_options(vectorforge_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Executable for testing and demonstration
add_executable(vectorforge_main
    src/main.cpp
)

target_link_libraries(vectorforge_main
    PRIVATE
        vectorforge_core
)

target_compile_features(vectorforge_main PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge_main PRIVATE /W4)
else()
    target_compile_options(vectorforge_main PRIVATE -Wall -Wextra -Wpedantic)
endif()
