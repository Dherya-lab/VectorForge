#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "vector.hpp"
#include "distance.hpp"
#include "simd_math.hpp"
#include "vector_index.hpp"
#include "hnsw_index.hpp"
#include "product_quantizer.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace vectorforge::python {

/**
 * @brief Unified Python wrapper for VectorForge vector search backends.
 */
class VectorForgeIndex {
public:
    VectorForgeIndex(
        std::size_t dimension,
        const std::string& backend = "hnsw",
        std::size_t m = 16,
        std::size_t ef_construction = 100,
        std::size_t ef_search = 50,
        uint32_t random_seed = 42
    ) : dimension_(dimension), backend_(backend), m_(m), ef_construction_(ef_construction), ef_search_(ef_search) {
        if (dimension == 0) {
            throw std::invalid_argument("Vector dimension must be greater than 0");
        }

        if (backend_ == "hnsw") {
            hnsw_index_ = std::make_unique<HNSWIndex>(HNSWConfig{
                .M = m,
                .ef_construction = ef_construction,
                .ef_search = ef_search,
                .random_seed = random_seed
            });
        } else if (backend_ == "bruteforce" || backend_ == "flat") {
            bf_index_ = std::make_unique<VectorIndex>(dimension);
        } else {
            throw std::invalid_argument("Unsupported search backend: '" + backend +
                                        "'. Supported backends: 'hnsw', 'bruteforce'");
        }
    }

    /**
     * @brief Inserts float32 vectors from a 1D (D,) or 2D (N, D) NumPy array.
     */
    std::size_t insert(py::array_t<float, py::array::c_style | py::array::forcecast> array) {
        py::buffer_info buf = array.request();
        const float* ptr = static_cast<const float*>(buf.ptr);
        std::size_t num_vectors = 0;

        if (buf.ndim == 1) {
            if (static_cast<std::size_t>(buf.shape[0]) != dimension_) {
                throw std::invalid_argument("Vector dimension mismatch: expected " +
                                            std::to_string(dimension_) + ", got " +
                                            std::to_string(buf.shape[0]));
            }
            num_vectors = 1;
            std::vector<float> vec_data(ptr, ptr + dimension_);
            Vector vec(std::move(vec_data));

            // Release GIL during insert
            py::gil_scoped_release release;
            if (hnsw_index_) {
                hnsw_index_->insert(std::move(vec));
            } else {
                bf_index_->insert(std::move(vec));
            }
        } else {
            const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
            const std::size_t d = static_cast<std::size_t>(buf.shape[1]);

            if (n == 0) {
                return 0;
            }
            if (d != dimension_) {
                throw std::invalid_argument("Vector dimension mismatch: expected " +
                                            std::to_string(dimension_) + ", got " +
                                            std::to_string(d));
            }
            num_vectors = n;

            // Zero-copy read from buffer into internal C++ index storage
            py::gil_scoped_release release;
            for (std::size_t i = 0; i < n; ++i) {
                const float* row_ptr = ptr + (i * d);
                std::vector<float> vec_data(row_ptr, row_ptr + d);
                Vector vec(std::move(vec_data));

                if (hnsw_index_) {
                    hnsw_index_->insert(std::move(vec));
                } else {
                    bf_index_->insert(std::move(vec));
                }
            }
        }

        return num_vectors;
    }

    /**
     * @brief Searches for top-k nearest neighbors of a query vector.
     */
    py::list search(py::array_t<float, py::array::c_style | py::array::forcecast> query_array, std::size_t k) const {
        if (k == 0) {
            throw std::invalid_argument("Search parameter k must be greater than 0");
        }

        py::buffer_info buf = query_array.request();

        if (buf.ndim == 1) {
            if (static_cast<std::size_t>(buf.shape[0]) != dimension_) {
                throw std::invalid_argument("Query dimension mismatch: expected " +
                                            std::to_string(dimension_) + ", got " +
                                            std::to_string(buf.shape[0]));
            }
        } else if (buf.ndim == 2 && buf.shape[0] == 1) {
            if (static_cast<std::size_t>(buf.shape[1]) != dimension_) {
                throw std::invalid_argument("Query dimension mismatch: expected " +
                                            std::to_string(dimension_) + ", got " +
                                            std::to_string(buf.shape[1]));
            }
        } else {
            throw std::invalid_argument("Query vector must be 1D array of shape (D,) or 2D array of shape (1, D)");
        }

        const float* query_ptr = static_cast<const float*>(buf.ptr);

        // Zero-copy span for querying
        std::vector<float> query_copy(query_ptr, query_ptr + dimension_);
        Vector query(std::move(query_copy));

        std::vector<SearchResult> results;
        {
            py::gil_scoped_release release;
            if (hnsw_index_) {
                results = hnsw_index_->search(query, k);
            } else {
                results = bf_index_->search(query, k);
            }
        }

        py::list py_results;
        for (const auto& r : results) {
            py::dict item;
            item["id"] = r.id;
            item["distance"] = r.distance;
            py_results.append(item);
        }

        return py_results;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        if (hnsw_index_) return hnsw_index_->size();
        if (bf_index_) return bf_index_->size();
        return 0;
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return dimension_;
    }

    [[nodiscard]] std::string backend() const noexcept {
        return backend_;
    }

    void set_ef_search(std::size_t ef_search) {
        if (ef_search == 0) {
            throw std::invalid_argument("ef_search must be greater than 0");
        }
        ef_search_ = ef_search;
        if (hnsw_index_) {
            hnsw_index_->set_ef_search(ef_search);
        }
    }

    py::dict tune(std::optional<std::size_t> ef_search,
                  std::optional<std::size_t> m,
                  std::optional<std::size_t> ef_construction) {
        if (ef_search.has_value()) {
            set_ef_search(ef_search.value());
        }
        if (m.has_value()) {
            m_ = m.value();
        }
        if (ef_construction.has_value()) {
            ef_construction_ = ef_construction.value();
        }

        py::dict result;
        result["ef_search"] = ef_search_;
        result["m"] = m_;
        result["ef_construction"] = ef_construction_;
        result["backend"] = backend_;
        return result;
    }

    [[nodiscard]] py::dict get_telemetry() const {
        py::dict tele;
        tele["size"] = size();
        tele["dimension"] = dimension_;
        tele["backend"] = backend_;

        if (hnsw_index_) {
            tele["max_level"] = hnsw_index_->max_level();
            auto ep = hnsw_index_->entry_point();
            if (ep.has_value()) {
                tele["entry_point"] = *ep;
            } else {
                tele["entry_point"] = py::none();
            }
            tele["ef_search"] = ef_search_;
            tele["ef_construction"] = ef_construction_;
            tele["m"] = m_;
        }

        const auto& cpu = simd::get_cpu_features();
        py::dict simd_dict;
        simd_dict["cpu_brand"] = cpu.cpu_brand;
        simd_dict["avx"] = cpu.has_avx;
        simd_dict["avx2"] = cpu.has_avx2;
        simd_dict["fma"] = cpu.has_fma;
        simd_dict["avx512f"] = cpu.has_avx512f;
        simd_dict["avx512dq"] = cpu.has_avx512dq;
        simd_dict["active_dispatch"] = cpu.supports_avx512() ? "AVX-512" :
                                      (cpu.supports_avx2() ? "AVX2 + FMA" : "Scalar Fallback");

        tele["simd"] = simd_dict;
        return tele;
    }

private:
    std::size_t dimension_{0};
    std::string backend_{"hnsw"};
    std::size_t m_{16};
    std::size_t ef_construction_{100};
    std::size_t ef_search_{50};

    std::unique_ptr<HNSWIndex> hnsw_index_;
    std::unique_ptr<VectorIndex> bf_index_;
};

py::dict get_simd_info() {
    const auto& cpu = simd::get_cpu_features();
    py::dict d;
    d["cpu_brand"] = cpu.cpu_brand;
    d["avx"] = cpu.has_avx;
    d["avx2"] = cpu.has_avx2;
    d["fma"] = cpu.has_fma;
    d["avx512f"] = cpu.has_avx512f;
    d["avx512dq"] = cpu.has_avx512dq;
    d["active_dispatch"] = cpu.supports_avx512() ? "AVX-512" :
                          (cpu.supports_avx2() ? "AVX2 + FMA" : "Scalar Fallback");
    return d;
}

} // namespace vectorforge::python

PYBIND11_MODULE(vectorforge, m) {
    m.doc() = "VectorForge high-performance C++20 vector search engine Python bindings";

    m.def("simd_info", &vectorforge::python::get_simd_info,
          "Returns CPU hardware capabilities and active SIMD distance dispatch kernel");

    py::class_<vectorforge::python::VectorForgeIndex>(m, "VectorForgeIndex")
        .def(py::init<std::size_t, const std::string&, std::size_t, std::size_t, std::size_t, uint32_t>(),
             py::arg("dimension"),
             py::arg("backend") = "hnsw",
             py::arg("m") = 16,
             py::arg("ef_construction") = 100,
             py::arg("ef_search") = 50,
             py::arg("random_seed") = 42,
             "Constructs a VectorForge index (backend: 'hnsw' or 'bruteforce')")
        .def("insert", &vectorforge::python::VectorForgeIndex::insert,
             py::arg("vectors"),
             "Inserts float32 vectors from a 1D (D,) or 2D (N, D) NumPy array")
        .def("search", &vectorforge::python::VectorForgeIndex::search,
             py::arg("query"),
             py::arg("k") = 10,
             "Searches for the k nearest neighbors for a float32 query vector")
        .def("size", &vectorforge::python::VectorForgeIndex::size,
             "Returns the number of vectors indexed")
        .def("dimension", &vectorforge::python::VectorForgeIndex::dimension,
             "Returns the vector dimensionality")
        .def("backend", &vectorforge::python::VectorForgeIndex::backend,
             "Returns the active search backend name")
        .def("set_ef_search", &vectorforge::python::VectorForgeIndex::set_ef_search,
             py::arg("ef_search"),
             "Sets runtime candidate beam width efSearch for HNSW")
        .def("tune", &vectorforge::python::VectorForgeIndex::tune,
             py::arg("ef_search") = py::none(),
             py::arg("m") = py::none(),
             py::arg("ef_construction") = py::none(),
             "Dynamically tunes index parameters")
        .def("get_telemetry", &vectorforge::python::VectorForgeIndex::get_telemetry,
             "Returns current telemetry and system metadata");
}
