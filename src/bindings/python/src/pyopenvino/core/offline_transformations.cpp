// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/offline_transformations.hpp"

#include <pybind11/stl.h>

#include <compress_quantize_weights.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/op/shape_of.hpp>
#include <pruning.hpp>
#include <transformations/common_optimizations/compress_float_constants.hpp>
#include <transformations/common_optimizations/fused_names_cleanup.hpp>
#include <transformations/resolve_names_collisions.hpp>
#include <transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp>
#include <transformations/common_optimizations/moc_legacy_transformations.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/flush_fp32_subnormals_to_zero.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/smart_reshape/smart_reshape.hpp>
#include <transformations/symbolic_transformations/symbolic_optimizations.hpp>

#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace py = pybind11;

static void unique_symbols_in_model(const std::shared_ptr<ov::Model>& m) {
    size_t num_shape_op_ops = 0;
    size_t num_shape_of_ops = 0;

    ov::pass::Validate().run_on_model(m);

    for (const auto& node : m->get_ops()) {
        if (node->get_rt_info().count("VP"))
            num_shape_op_ops++;
        if (ov::is_type<ov::op::v0::ShapeOf>(node) || ov::is_type<ov::op::v3::ShapeOf>(node))
            num_shape_of_ops++;
    }
    std::cout << "# ops: " << m->get_ops().size() << std::endl;
    std::cout << "# shape sub-graph ops: " << num_shape_op_ops << std::endl;
    std::cout << "# shape of ops: " << num_shape_of_ops << std::endl;
}

void regmodule_offline_transformations(py::module m) {
    py::module m_offline_transformations =
        m.def_submodule("_offline_transformations", "Offline transformations module");
    m_offline_transformations.doc() =
        "openvino._offline_transformations is a private module contains different offline passes.";

    m_offline_transformations.def(
        "apply_moc_transformations",
        [](std::shared_ptr<ov::Model> model, bool cf, bool smart_reshape) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::SymbolicOptimizations>();
            manager.register_pass<ov::pass::ResolveNameCollisions>(true);
            manager.run_passes(model);
        },
        py::arg("model"),
        py::arg("cf"),
        py::arg("smart_reshape") = false);

    m_offline_transformations.def(
        "apply_moc_legacy_transformations",
        [](std::shared_ptr<ov::Model> model, const std::vector<std::string>& params_with_custom_types) {
            unique_symbols_in_model(model);
        },
        py::arg("model"),
        py::arg("params_with_custom_types"));

    m_offline_transformations.def(
        "apply_low_latency_transformation",
        [](std::shared_ptr<ov::Model> model, bool use_const_initializer = true) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::LowLatency2>(use_const_initializer);
            manager.run_passes(model);
        },
        py::arg("model"),
        py::arg("use_const_initializer") = true);

    m_offline_transformations.def(
        "apply_pruning_transformation",
        [](std::shared_ptr<ov::Model> model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::Pruning>();
            manager.run_passes(model);
        },
        py::arg("model"));

    m_offline_transformations.def(
        "apply_make_stateful_transformation",
        [](std::shared_ptr<ov::Model> model, const std::map<std::string, std::string>& param_res_names) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MakeStateful>(param_res_names);
            manager.run_passes(model);
        },
        py::arg("model"),
        py::arg("param_res_names"));

    m_offline_transformations.def(
        "apply_make_stateful_transformation",
        [](std::shared_ptr<ov::Model> model, const ov::pass::MakeStateful::ParamResPairs& pairs_to_replace) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MakeStateful>(pairs_to_replace);
            manager.run_passes(model);
        },
        py::arg("model"),
        py::arg("pairs_to_replace"));

    m_offline_transformations.def(
        "compress_model_transformation",
        [](std::shared_ptr<ov::Model> model) {
            bool postponed = false;
            return ov::pass::compress_model_to_f16(model, postponed);
        },
        py::arg("model"));

    m_offline_transformations.def(
        "compress_quantize_weights_transformation",
        [](std::shared_ptr<ov::Model> model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::CompressQuantizeWeights>();
            manager.run_passes(model);
        },
        py::arg("model"));

    m_offline_transformations.def(
        "convert_sequence_to_tensor_iterator_transformation",
        [](std::shared_ptr<ov::Model> model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
            manager.run_passes(model);
        },
        py::arg("model"));

    m_offline_transformations.def(
        "apply_fused_names_cleanup",
        [](std::shared_ptr<ov::Model> model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::FusedNamesCleanup>();
            manager.run_passes(model);
        },
        py::arg("model"));

    m_offline_transformations.def(
        "paged_attention_transformation",
        [](std::shared_ptr<ov::Model> model) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::SDPAToPagedAttention>();
            manager.run_passes(model);
        },
        py::arg("model"));
}
