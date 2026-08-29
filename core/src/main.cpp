#include "vector.hpp"

#include <iostream>

int main() {
    std::cout << "VectorForge Phase 0" << std::endl;

    // Create a small vector embedding
    vectorforge::Vector vec{0.1f, 0.2f, 0.3f, 0.4f};

    // Print dimensionality
    std::cout << "Dimension: " << vec.dimension() << std::endl;

    // Print values
    std::cout << "Values: ";
    for (std::size_t i = 0; i < vec.dimension(); ++i) {
        std::cout << vec[i] << (i + 1 < vec.dimension() ? " " : "");
    }
    std::cout << std::endl;

    return 0;
}
