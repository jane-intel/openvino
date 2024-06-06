// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/nop_broadcast.hpp"

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::symbol::util;

namespace {
shared_ptr<Node> broadcast_label(const OutputVector& inputs) {
    return ov::pass::pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>(inputs, [](Output<Node> output) {
        const auto& op = output.get_node_shared_ptr();
        auto data_rank = op->get_input_partial_shape(0).rank();
        auto new_shape_shape = op->get_input_partial_shape(1);
        return data_rank.is_static() && new_shape_shape.is_static() && data_rank == new_shape_shape[0];
    });
}
}  // namespace

ov::pass::NopBroadcast::NopBroadcast() {
    MATCHER_SCOPE(NopBroadcast);
    auto data_label = pattern::any_input(pattern::has_static_rank());

    auto shape_label = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>();
    auto ones = INT_CONSTANT_WITH_PREDICATE(std::all_of(value.begin(), value.end(), cmp::Equal<int64_t>(1)));
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_label, ones});

    auto broadcast_3_ins = broadcast_label({data_label, maximum, pattern::any_input()});
    auto broadcast_2_ins = broadcast_label({data_label, maximum});
    auto broadcast = make_shared<pattern::op::Or>(OutputVector{broadcast_2_ins, broadcast_3_ins});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        auto data = vm.at(data_label);
        auto shape = vm.at(shape_label);

        ov::TensorSymbol data_symbols, shape_symbols;
        if (!get_symbols(data.get_partial_shape(), data_symbols) || !get_symbols(shape, shape_symbols) ||
            !are_unique_and_equal_symbols(data_symbols, shape_symbols))
            return false;
        return ov::replace_output_update_name(m.get_match_root(), data);
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::NopBroadcastSubGraph::NopBroadcastSubGraph() {
    MATCHER_SCOPE(NopBroadcastSubGraph);
    auto data_label = pattern::any_input(pattern::has_static_rank());

    auto shape_label = pattern::wrap_type<op::v0::ShapeOf, op::v3::ShapeOf>({data_label});
    auto tensor_label = pattern::any_input(pattern::has_static_shape());
    auto maximum = pattern::wrap_type<op::v1::Maximum>({shape_label, tensor_label});

    auto broadcast_3_ins =
        pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({data_label, maximum, pattern::any_input()});
    auto broadcast_2_ins = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({data_label, maximum});
    auto broadcast = make_shared<pattern::op::Or>(OutputVector{broadcast_2_ins, broadcast_3_ins});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        ov::PartialShape data_shape = vm.at(data_label).get_partial_shape();
        ov::PartialShape tensor_as_shape;
        ov::util::evaluate_as_partial_shape(vm.at(tensor_label), tensor_as_shape);
        if (data_shape.size() != tensor_as_shape.size())
            return false;
        for (size_t i = 0; i < data_shape.size(); ++i) {
            if (tensor_as_shape[i] == 1)
                continue;
            if (ov::symbol::are_equal(tensor_as_shape[i].get_symbol(), data_shape[i].get_symbol()))
                continue;
            return false;
        }
        return ov::replace_output_update_name(m.get_match_root(), vm.at(data_label));
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

static bool tensor_non_negative(const ov::Tensor& tensor) {
    const auto& lower =
        std::make_shared<op::v0::Parameter>(tensor.get_element_type(), ov::PartialShape(tensor.get_shape()));
    const auto& gr_eq =
        std::make_shared<op::v1::GreaterEqual>(lower, op::v0::Constant::create(tensor.get_element_type(), {}, {0}));
    auto axes_vector = std::vector<int64_t>(tensor.get_shape().size());
    std::iota(axes_vector.begin(), axes_vector.end(), 0);
    const auto axes = op::v0::Constant::create(element::i64, {axes_vector.size()}, axes_vector);
    const auto& result = std::make_shared<op::v1::ReduceLogicalAnd>(gr_eq, axes);
    ov::Model m(OutputVector{result}, ParameterVector{lower});
    ov::TensorVector output(1);
    m.evaluate(output, {tensor});
    return op::v0::Constant(output[0]).cast_vector<bool>()[0];
}

ov::pass::NopBroadcastSubGraph2::NopBroadcastSubGraph2() {
    MATCHER_SCOPE(NopBroadcastSubGraph2);

    auto data_label = pattern::any_input([](Output<Node> out) {
        return out.get_partial_shape().is_static() && ov::shape_size(out.get_shape()) > 0;
    });

    auto tensor_label = pattern::any_input();
    auto ones = INT_CONSTANT_WITH_PREDICATE(std::all_of(value.begin(), value.end(), cmp::Equal<int64_t>(1)));

    auto maximum = pattern::wrap_type<op::v1::Maximum>({tensor_label, ones});

    auto broadcast_3_ins =
        pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({data_label, maximum, pattern::any_input()});
    auto broadcast_2_ins = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast>({data_label, maximum});
    auto broadcast = make_shared<pattern::op::Or>(OutputVector{broadcast_2_ins, broadcast_3_ins});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& vm = m.get_pattern_value_map();
        Tensor lb, ub;
        std::tie(lb, ub) = evaluate_both_bounds(vm.at(tensor_label));
        return tensor_non_negative(lb) && ov::replace_output_update_name(vm.at(maximum), vm.at(tensor_label));
    };

    auto m = std::make_shared<pattern::Matcher>(broadcast, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
