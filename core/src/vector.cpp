#include "vector.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace vectorforge {

Vector::Vector(std::size_t dimension, float initial_value)
    : data_(dimension, initial_value) {}

Vector::Vector(std::initializer_list<float> values)
    : data_(values) {}

Vector::Vector(std::vector<float> values)
    : data_(std::move(values)) {}

Vector::Vector(const float* data, std::size_t size)
    : data_(data, data + size) {}

Vector::Vector(std::span<const float> values)
    : data_(values.begin(), values.end()) {}

std::size_t Vector::dimension() const noexcept {
    return data_.size();
}

std::size_t Vector::size() const noexcept {
    return data_.size();
}

bool Vector::empty() const noexcept {
    return data_.empty();
}

float Vector::operator[](std::size_t index) const noexcept {
    return data_[index];
}

float& Vector::operator[](std::size_t index) noexcept {
    return data_[index];
}

float Vector::at(std::size_t index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
}

float& Vector::at(std::size_t index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
}

const float* Vector::data() const noexcept {
    return data_.data();
}

float* Vector::data() noexcept {
    return data_.data();
}

const std::vector<float>& Vector::values() const noexcept {
    return data_;
}

std::ostream& operator<<(std::ostream& os, const Vector& vec) {
    os << "[";
    for (std::size_t i = 0; i < vec.dimension(); ++i) {
        os << vec[i];
        if (i + 1 < vec.dimension()) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

} // namespace vectorforge
