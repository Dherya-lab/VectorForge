#pragma once

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <span>
#include <vector>

namespace vectorforge {

/**
 * @brief Represents a dense floating-point vector/embedding.
 */
class Vector {
public:
    /// Default constructor creates an empty vector
    Vector() = default;

    /// Constructs a vector with a given dimension, optionally filled with an initial value
    explicit Vector(std::size_t dimension, float initial_value = 0.0f);

    /// Constructs a vector from an initializer list of floats
    Vector(std::initializer_list<float> values);

    /// Constructs a vector by copying or moving a std::vector<float>
    explicit Vector(std::vector<float> values);

    /// Constructs a vector from a raw pointer and size
    Vector(const float* data, std::size_t size);

    /// Constructs a vector from a std::span
    explicit Vector(std::span<const float> values);

    /// Returns the dimensionality (number of elements) of the vector
    [[nodiscard]] std::size_t dimension() const noexcept;

    /// Alias for dimension() for STL compatibility
    [[nodiscard]] std::size_t size() const noexcept;

    /// Returns true if the vector contains no elements
    [[nodiscard]] bool empty() const noexcept;

    /// Unchecked element access (const)
    [[nodiscard]] float operator[](std::size_t index) const noexcept;

    /// Unchecked element access (non-const)
    [[nodiscard]] float& operator[](std::size_t index) noexcept;

    /// Bounds-checked element access (const)
    [[nodiscard]] float at(std::size_t index) const;

    /// Bounds-checked element access (non-const)
    [[nodiscard]] float& at(std::size_t index);

    /// Direct pointer to underlying contiguous float buffer (const)
    [[nodiscard]] const float* data() const noexcept;

    /// Direct pointer to underlying contiguous float buffer (non-const)
    [[nodiscard]] float* data() noexcept;

    /// Direct reference to underlying vector data (const)
    [[nodiscard]] const std::vector<float>& values() const noexcept;

    // Iterators for range-based for loops
    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }
    [[nodiscard]] auto cbegin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] auto cend() const noexcept { return data_.cend(); }

private:
    std::vector<float> data_;
};

/// Stream output operator for clean printing
std::ostream& operator<<(std::ostream& os, const Vector& vec);

} // namespace vectorforge
