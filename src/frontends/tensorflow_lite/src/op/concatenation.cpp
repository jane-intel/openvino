// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector concatenation(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    const std::map<std::string, ov::Any> attrs{
        {"axis", static_cast<int64_t>(decoder->get_attribute(&tflite::ConcatenationOptions::axis))},
    };
    auto output = attribute_helper(
            node, attrs, ov::frontend::tensorflow::op::translate_concat_op, "tflite::CONCATENATION", true);
    del_output_names(output);
    get_activation(output, node, EnumNameActivationFunctionType(
            decoder->get_attribute(&tflite::ConcatenationOptions::fused_activation_function)));
    del_output_names(output);
    output[0].get_node_shared_ptr()->set_friendly_name(decoder->get_op_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
