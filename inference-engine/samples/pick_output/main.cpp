// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/classification_results.h>

#include <inference_engine.hpp>
#include <openvino/opsets/opset8.hpp>
#include <iterator>
#include <openvino/pass/visualize_tree.hpp>

using namespace InferenceEngine;


int main(int argc, char* argv[]) {
    // ------------------------------ Parsing and validation of input arguments
    // ---------------------------------
    if (argc != 2) {
        std::cout << "Usage : " << argv[0] << " <path_to_model>" << std::endl;
        return -1;
    }

    const auto input_model{argv[1]};
    Core ie;
    CNNNetwork network = ie.ReadNetwork(input_model);

    auto get_node_by_name = [](std::shared_ptr<ov::Function> f, const std::string& name) -> std::shared_ptr<ov::Node> {
        std::shared_ptr<ov::Node> node = nullptr;
        for (const auto& op : f->get_ordered_ops()) {
            if (op->get_friendly_name() == name) {
                node = op;
                break;
            }
        }
        OPENVINO_ASSERT(node != nullptr, "Couldn't find node with name " + name);
        return node;
    };

    auto graph = network.getFunction();
    auto parameters = graph->get_parameters();
    auto results = graph->get_results();

    for (const auto& out : results)
        std::cout << "OUT: " << out << std::endl;

    auto scale_parameter = std::make_shared<ov::opset8::Parameter>(ov::element::u8, ov::Shape{});
    scale_parameter->set_friendly_name("scale_switch")
    graph->add_parameters({scale_parameter});
    // remove results and pick a name for it

    ////////////// If scale_parameter==1
    /// condition
    auto scale_eq_one = std::make_shared<ov::opset8::Equal>(
            scale_parameter, ov::opset8::Constant::create(ov::element::u8, ov::Shape{}, {1}));

    std::shared_ptr<ov::Node> fq = get_node_by_name(graph, "tl_unet1x2x4x/Decoder_2x/Conv2D/fq_input_0");
    std::shared_ptr<ov::Node> conv = get_node_by_name(graph, "tl_unet1x2x4x/Decoder_2x/Conv2D");
    std::shared_ptr<ov::Node> out_1x = get_node_by_name(graph, "tl_unet1x2x4x/resize_images/ResizeBilinear/fq_input_0");
    std::shared_ptr<ov::Node> out_1x_fq = get_node_by_name(graph, "tl_unet1x2x4x/Decoder_4x/Conv2D/fq_input_0");
    std::shared_ptr<ov::Node> out_2x = get_node_by_name(graph, "tl_unet1x2x4x/resize_images_2/ResizeBilinear");
    std::shared_ptr<ov::Node> out_2x_fq = get_node_by_name(graph, "tl_unet1x2x4x/Decoder_4x/Conv2D");
    std::shared_ptr<ov::Node> out_4x = get_node_by_name(graph, "tl_unet1x2x4x/out4x/add_2");

    /// then (passthrough of the first output)
    auto then_param = std::make_shared<ov::opset8::Parameter>(results[0]->get_output_element_type(0), results[0]->get_output_partial_shape(0));
    auto then_fq_param = std::make_shared<ov::opset8::Parameter>(fq->get_output_element_type(0), fq->get_output_partial_shape(0));
    auto then_result = std::make_shared<ov::opset8::Result>(then_param);
    auto then_fq_result = std::make_shared<ov::opset8::Result>(then_fq_param);

    auto then_function = std::make_shared<ov::Function>(
            ov::ResultVector{then_result, then_fq_result}, ov::ParameterVector{then_param, then_fq_param});
    // else (second stage of the upscaling -- from FQ to the second results add)
    auto else_param = std::make_shared<ov::opset8::Parameter>(results[0]->get_output_element_type(0), results[0]->get_output_partial_shape(0));
    auto else_fq_param = std::make_shared<ov::opset8::Parameter>(fq->get_output_element_type(0), fq->get_output_partial_shape(0));

    out_1x->input(0).replace_source_output(else_param->output(0));
    conv->input(0).replace_source_output(else_fq_param->output(0));

    auto else_result = std::make_shared<ov::opset8::Result>(out_2x->input_value(0));
    auto else_fq_result = std::make_shared<ov::opset8::Result>(out_1x_fq);
    auto else_function = std::make_shared<ov::Function>(
            ov::ResultVector{else_result, else_fq_result}, ov::ParameterVector{else_param, else_fq_param});


    auto first_if = std::make_shared<ov::opset8::If>(scale_eq_one);
    first_if->set_then_body(then_function);
    first_if->set_else_body(else_function);

    first_if->set_input(results[0]->input_value(0), then_param, else_param);
    first_if->set_input(fq->output(0), then_fq_param, else_fq_param);

    first_if->set_output(then_result, else_result);
    first_if->set_output(then_fq_result, else_fq_result);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto scale_less_three = std::make_shared<ov::opset8::Less>(
            scale_parameter, ov::opset8::Constant::create(ov::element::u8, ov::Shape{}, {3}));

    auto then_param_1 = std::make_shared<ov::opset8::Parameter>(first_if->get_output_element_type(0), first_if->get_output_partial_shape(0));
    auto then_fq_param_1 = std::make_shared<ov::opset8::Parameter>(first_if->get_output_element_type(1), first_if->get_output_partial_shape(1));
    auto then_result_1 = std::make_shared<ov::opset8::Result>(then_param_1);

    auto then_function_1 = std::make_shared<ov::Function>(
            ov::ResultVector{then_result_1}, ov::ParameterVector{then_param_1, then_fq_param_1});
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto else_param_1 = std::make_shared<ov::opset8::Parameter>(first_if->get_output_element_type(0), first_if->get_output_partial_shape(0));
    auto else_fq_param_1 = std::make_shared<ov::opset8::Parameter>(first_if->get_output_element_type(1), first_if->get_output_partial_shape(1));

    out_2x->input(0).replace_source_output(else_param_1);
    out_2x_fq->input(0).replace_source_output(else_fq_param_1);

    auto else_result_1 = std::make_shared<ov::opset8::Result>(out_4x);
    auto else_function_1 = std::make_shared<ov::Function>(
            ov::ResultVector{else_result_1}, ov::ParameterVector{else_param_1, else_fq_param_1});

    auto second_if = std::make_shared<ov::opset8::If>(scale_less_three);
    second_if->set_then_body(then_function_1);
    second_if->set_else_body(else_function_1);

    second_if->set_input(first_if->output(0), then_param_1, else_param_1);
    second_if->set_input(first_if->output(1), then_fq_param_1, else_fq_param_1);
    second_if->set_output(then_result_1, else_result_1);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////


    ov::pass::VisualizeTree("/localdisk/orig.svg").run_on_function(graph);


    auto single_result = std::make_shared<ov::opset8::Result>(second_if->output(0));
    single_result->set_friendly_name("single_output");
    graph->add_results(ov::ResultVector{single_result});
    graph->remove_result(results[0]);
    graph->remove_result(results[1]);
    graph->remove_result(results[2]);


    ov::pass::VisualizeTree("/localdisk/then.svg").run_on_function(then_function);
    ov::pass::VisualizeTree("/localdisk/then_1.svg").run_on_function(then_function_1);
    ov::pass::VisualizeTree("/localdisk/else.svg").run_on_function(else_function);
    ov::pass::VisualizeTree("/localdisk/else_1.svg").run_on_function(else_function_1);
    ov::pass::VisualizeTree("/localdisk/orig_modified.svg").run_on_function(graph);

    std::cout << "NEW IN with possible values 1 for 1x, 2 for 2x, 4 for 4x: " << scale_parameter << std::endl;
    std::cout << "NEW OUT: " << single_result << std::endl;

    CNNNetwork(graph).serialize("/localdisk/3_output_to_one/single_output.xml", "/localdisk/3_output_to_one/single_output.bin");
    std::cout << "Success" << std::endl;
    return 0;
}