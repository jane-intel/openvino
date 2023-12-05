// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/bitwise_and.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
BitwiseAnd::BitwiseAnd(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseBitwise(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> BitwiseAnd::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_BitwiseAnd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BitwiseAnd>(new_args[0], new_args[1], get_autob());
}

}  // namespace v13
}  // namespace op
}  // namespace ov
