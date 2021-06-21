// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fully_connected.hpp"
#include "op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <transformations/smart_reshape/utils.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapeFullyConnected, "ReshapeFullyConnected", 0);

MKLDNNPlugin::ReshapeFullyConnected::ReshapeFullyConnected() {
    ngraph::OutputVector twoInputs = {ngraph::pattern::any_input(ngraph::pattern::has_static_rank()), ngraph::pattern::any_input()};
    ngraph::OutputVector threeInputs = {ngraph::pattern::any_input(ngraph::pattern::has_static_rank()), ngraph::pattern::any_input(),
                                        ngraph::pattern::any_input()};
    auto fcTwoInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(twoInputs, ngraph::pattern::has_static_rank());
    auto fcThreeInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(threeInputs, ngraph::pattern::has_static_rank());
    const auto fcTwoOrThreeInputs = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fcTwoInputs, fcThreeInputs});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto fc = std::dynamic_pointer_cast<MKLDNNPlugin::FullyConnectedNode> (m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        auto input_shape = fc->get_input_partial_shape(0);
        auto input_rank = input_shape.rank().get_length();
        auto output_shape = fc->get_output_partial_shape(0);

        if (input_rank <= 2) {
            return false;
        }

        ngraph::NodeVector new_ops;
        auto K_idx = static_cast<size_t>(input_rank - 1);
        auto K_node = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(fc->input_value(0), {K_idx});

        ngraph::OutputVector shape_dims = {ngraph::opset1::Constant::create(K_node->get_output_element_type(0), {1}, {-1}), K_node};
        auto reshape_shape = std::make_shared<ngraph::opset1::Concat>(shape_dims, 0);
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(fc->input_value(0), reshape_shape, false);
        if (reshape->get_output_partial_shape(0).rank().is_dynamic())
            return false;
        new_ops.push_back(reshape);

        reshape->set_friendly_name(fc->get_friendly_name() + "/Reshape");

        // Calculate output shape for new FullyConnected layer
        // [I, K] * [O, K] = [I, O]
        auto I = reshape->get_output_partial_shape(0)[0];
        auto O = fc->get_input_partial_shape(1)[0];
        ngraph::PartialShape output_shape_new{I, O};

        std::shared_ptr<ngraph::Node> fc_new;
        if (fc->get_input_size() == 2) {
            fc_new = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape,
                                                                        fc->input_value(1),
                                                                        output_shape_new,
                                                                        fc->get_output_type());
        } else if (fc->get_input_size() == 3) {
            fc_new = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape,
                                                                        fc->input_value(1),
                                                                        fc->input_value(2),
                                                                        output_shape_new,
                                                                        fc->get_output_type());
        } else {
            return false;
        }
        new_ops.push_back(fc_new);

        if (output_shape != output_shape_new) {
            auto I_idxs = std::vector<size_t>(input_rank - 1);
            std::iota(I_idxs.begin(), I_idxs.end(), 0);
            auto I_node = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(fc->input_value(0), {I_idxs});
            auto O_node = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(fc->input_value(1), {0});
            auto reshape_back_shape = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{I_node, O_node}, 0);
            auto reshape_output = std::make_shared<ngraph::opset1::Reshape>(fc_new, reshape_back_shape, false);
            new_ops.push_back(reshape_output);
            reshape_output->set_friendly_name(fc->get_friendly_name());
            fc_new->set_friendly_name(fc->get_friendly_name() + "/FC");
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, reshape_output);
        } else {
            fc_new->set_friendly_name(fc->get_friendly_name());
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, fc_new);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fcTwoOrThreeInputs, "ReshapeFullyConnected");
    this->register_matcher(m, callback);
}
