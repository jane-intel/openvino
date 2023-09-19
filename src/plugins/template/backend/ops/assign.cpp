// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluate_node.hpp"

bool evaluate(const std::shared_ptr<ov::op::v3::Assign>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    outputs[0].set_shape(inputs[0].get_shape());
    inputs[0].copy_to(outputs[0]);
    return true;
}

template <>
bool evaluate_node<ov::op::v3::Assign>(std::shared_ptr<ov::Node> node,
                                       ov::TensorVector& outputs,
                                       const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    return evaluate(ov::as_type_ptr<ov::op::v3::Assign>(node), outputs, inputs);
}
