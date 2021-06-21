// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "itt.hpp"
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SharedShapeOf, "SharedShapeOf", 0);

bool ngraph::pass::SharedShapeOf::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(SharedShapeOf);
    bool graph_rewritten = false;

    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::opset1::ShapeOf>>> source_to_shape_of;
    std::map<ngraph::Output<Node>, std::vector<std::shared_ptr<ngraph::opset3::ShapeOf>>> source_to_shape_of_3;
    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto shape_of = std::dynamic_pointer_cast<ngraph::opset1::ShapeOf>(node)) {
            source_to_shape_of[shape_of->input_value(0)].push_back(shape_of);
        } else if (auto shape_of_3 = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(node)) {
            source_to_shape_of_3[shape_of_3->input_value(0)].push_back(shape_of_3);
        }
    }

    for (auto& pair : source_to_shape_of) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id()) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    for (auto& pair : source_to_shape_of_3) {
        if (pair.second.size() < 2)
            continue;
        auto root_ss = pair.second[0];
        for (auto& child_ss : pair.second) {
            if (root_ss->get_instance_id() != child_ss->get_instance_id() && root_ss->get_output_type() == child_ss->get_output_type()) {
                graph_rewritten |= replace_output_update_name(child_ss->output(0), root_ss->output(0));
            }
        }
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupedGatherElimination, "GroupedGatherElimination", 0);

bool ngraph::pass::GroupedGatherElimination::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedGatherElimination);
    bool graph_rewritten = false;

    for (const auto & node : f->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                graph_rewritten |= run_on_function(sub_graph);
            }
        }
        if (auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(node)) {
            if (concat->get_output_partial_shape(0).rank() != 1)
                continue;

            std::map<int64_t, std::shared_ptr<Node>> idx_to_gather;
            for (const auto& input : concat->inputs()) {
                if (auto gather = std::dynamic_pointer_cast<ngraph::opset1::Gather>(input.get_source_output().get_node_shared_ptr()))
                    idx_to_gather[input.get_index()] = gather;
            }
            if (idx_to_gather.size() < 2)
                continue;

            const auto& source = idx_to_gather.begin()->second->input_value(0);
            bool indices_are_consequtive = true;
            int64_t first_idx = 0, last_idx = -1;
            OutputVector indices;
            for (const auto& item : idx_to_gather) {
                indices_are_consequtive &= last_idx == -1 || last_idx + 1 == item.first;
                if (!indices_are_consequtive)
                    break;
                if (last_idx == -1)
                    first_idx = item.first;
                last_idx = item.first;
                indices.push_back(item.second->input_value(1));
            }
            if (indices_are_consequtive && last_idx != -1) {
                auto new_gather = std::make_shared<ngraph::opset1::Gather>(
                         source,
                         std::make_shared<opset1::Concat>(indices, 0),
                         ngraph::opset1::Constant::create(element::i32, {}, {0}));

                auto inputs = concat->input_values();
                inputs.erase(inputs.begin() + first_idx, inputs.begin() + last_idx + 1);
                inputs.insert(inputs.begin() + first_idx, new_gather);
                auto new_concat = std::make_shared<opset1::Concat>(inputs, 0);

                new_concat->set_friendly_name(concat->get_friendly_name());
                ngraph::copy_runtime_info(concat, new_concat);
                ngraph::replace_node(concat, new_concat);
                graph_rewritten = true;
            }
        }
    }
    return graph_rewritten;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SimplifyShapeOfSubGraph, "SimplifyShapeOfSubGraph", 0);

bool ngraph::pass::SimplifyShapeOfSubGraph::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(GroupedGatherElimination);
    ngraph::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ngraph::pass::EliminateGatherUnsqueeze>();
    manager.register_pass<ngraph::pass::SharedShapeOf>();
    manager.register_pass<ngraph::pass::GroupedGatherElimination>();
    manager.register_pass<ngraph::pass::Validate>();
    manager.run_passes(f);
    return false;
}
