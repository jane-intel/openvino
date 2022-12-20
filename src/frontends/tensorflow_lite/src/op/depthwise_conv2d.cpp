// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include "op_translation_utils.hpp"


using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector depthwise_conv2d(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    auto decoder_for_tf_translator = get_conv_decoder_map<tflite::DepthwiseConv2DOptions>("DepthwiseConv2DOptions", "DepthwiseConv2dNative", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2, "Unexpected number of input in node of type=", node.get_op_type(), " name=", node.get_name());
    OutputVector output;
    get_conv(output, node, decoder_for_tf_translator, &ov::frontend::tensorflow::op::translate_depthwise_conv_2d_native_op);
    get_bias(output, node, decoder_for_tf_translator);
    get_activation(output, node, decoder_for_tf_translator);
    const auto& decoder = dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr, "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    output[0].set_names({decoder->get_output_tensor_name(0)});
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov



