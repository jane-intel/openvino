// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>

#include "input_model.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {

using namespace ov::op;

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::map<std::string, CreatorFunction>& translator_map,
                                   const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_telemetry(telemetry),
      m_ov_model(nullptr) {}

TranslateSession::~TranslateSession() {
    if (m_telemetry) {
        // Send statistics
        for (const auto& op : m_op_statistics) {
            m_telemetry->send_event("op_count", "jax_" + op.first, static_cast<int>(op.second));
        }
    }
}

std::shared_ptr<ov::Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<ov::Model> TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model) {
    auto jax_model = std::dynamic_pointer_cast<jax::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(jax_model != nullptr, "Invalid input model");
    auto model = convert_jax_model(jax_model->m_model_decoder, jax_model);
    // First delete tensor indexes from outputs then resolve input names,
    // otherwise Parameter->Result will fail
    for (auto& result : model->get_results()) {
        auto tensor_desc = result->input_value(0);
        auto names = tensor_desc.get_names();
        if (!names.empty()) {
            auto tensor_idx = decode_tensor_name(tensor_desc);
            if (names.erase(std::to_string(tensor_idx))) {
                tensor_desc.set_names(names);
            }
        }
    }
    // Set input tensor names to be equal to signature name saved in friendly name
    for (auto& param : model->get_parameters()) {
        if (param->get_friendly_name() != param->get_name()) {
            // get_name is autogenerated name, we need to make sure that this
            // parameter was named by frontend
            param->output(0).set_names({param->get_friendly_name()});
        }
    }
    return model;
}

std::shared_ptr<Model> TranslateSession::convert_jax_model(std::shared_ptr<JaxDecoder> jax_model,
                                                           const std::shared_ptr<jax::InputModel>& input_model) {
    std::shared_ptr<Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        auto parameters = std::make_shared<ParameterVector>();
        auto tensor_map = std::make_shared<TensorMap>();  // tensor map of the current context

        FRONT_END_GENERAL_CHECK(input_model,
                                "Got null input model in JAX frontend TranslateSession::convert_jax_model.");

        // When we have input model we should use its inputs order to create
        // Parameters We use m_inputs instead of get_inputs() because latter
        // doesn't have "self" input
        for (auto& input_p : input_model->get_inputs()) {
            auto jax_place = std::dynamic_pointer_cast<jax::Place>(input_p);
            FRONT_END_GENERAL_CHECK(jax_place, "Only place produced by Jax Frontend is supported.");
            auto tensor_id = jax_place->get_tensor_index();
            element::Type type = jax_place->get_element_type();
            PartialShape pshape = jax_place->get_partial_shape();
            auto parameter = std::make_shared<v0::Parameter>(type, pshape);
            if (jax_place->get_names().size() > 0)
                parameter->set_friendly_name(jax_place->get_names().at(0));
            encode_tensor_name(parameter->output(0), tensor_id);
            parameters->push_back(parameter);
            (*tensor_map)[tensor_id] = parameter;
        }

        auto node_visitor = [&](std::shared_ptr<JaxDecoder> node) {
            // Get named params, if there's any.
            std::unordered_map<std::string, size_t> param_name_to_id;
            auto param_names = node->get_param_names();
            for (auto name : param_names) {
                auto param_tensor_id = node->get_named_param(name);
                if (tensor_map->find(param_tensor_id) == tensor_map->end()) {
                    auto constant = node->get_named_param_as_constant(name);
                    // TODO: maybe allow list constant here if needed?
                    FRONT_END_GENERAL_CHECK(constant.size() == 1, "Constant should have only one value yet.");
                    (*tensor_map)[param_tensor_id] = constant[0];
                }
                param_name_to_id[name] = param_tensor_id;
            }

            auto context = NodeContext(node, tensor_map, parameters, param_name_to_id, this);
            // Add op type in the statistics
            m_op_statistics[context.get_op_type()]++;
            auto converted_outputs = convert_node(context);

            const auto& fw_outputs = node->outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after
            // conversion compared to jax ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          context.get_op_type(),
                                          " outputs greater than number of converted outputs, which are",
                                          fw_outputs.size(),
                                          " and ",
                                          converted_outputs.size(),
                                          " respectively.");

            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                FRONT_END_GENERAL_CHECK(tensor_map->find(fw_tensor_id) == tensor_map->end(),
                                        "Duplicated producer for Jax value with unique ID: ",
                                        fw_tensor_id);
#ifdef ENABLE_OPENVINO_DEBUG
                auto out_type = context.get_output_type(i);
                if (out_type.is<element::Type>()) {
                    if (!converted_outputs[i].get_element_type().compatible(out_type.as<element::Type>())) {
                        OPENVINO_DEBUG("[WARNING] Produced output type for operation ",
                                       context.get_op_type(),
                                       " for tensor id: ",
                                       fw_tensor_id,
                                       " is incompatible: produced ",
                                       converted_outputs[i].get_element_type(),
                                       " vs ",
                                       out_type.as<element::Type>());
                    }
                }
#endif
                (*tensor_map)[fw_tensor_id] = converted_outputs[i];
                encode_tensor_name(converted_outputs[i], fw_tensor_id, {node->get_output_name(i)});
            }
        };

        jax_model->visit_subgraph(node_visitor);

        ResultVector results;
        // For the case when we have InputModel we need to have same order as its
        // outputs
        for (auto& output_p : input_model->get_outputs()) {
            auto jax_place = std::dynamic_pointer_cast<jax::Place>(output_p);
            FRONT_END_GENERAL_CHECK(jax_place, "Only place produced by Jax Frontend is supported.");
            auto tensor_id = jax_place->get_tensor_index();
            auto ov_output = tensor_map->at(tensor_id);
            FRONT_END_GENERAL_CHECK(ov_output.get_names().size() > 0,
                                    "Tensor doesn't have name, while it should have name: ",
                                    tensor_id);
            auto result = std::make_shared<v0::Result>(ov_output);
            results.push_back(result);
        }

        resulting_model = std::make_shared<Model>(results, *parameters);
    }

    return resulting_model;
}

OutputVector TranslateSession::convert_node(const NodeContext& context) {
    std::string exception;
    try {
        auto it = m_translator_map.find(context.get_op_type());
        if (it != m_translator_map.end()) {
            return it->second(context);
        }
        OPENVINO_DEBUG("No translator found for: ", context.get_op_type(), "\n");
    } catch (std::exception& e) {
        exception = e.what();
        if (m_telemetry) {
            auto cropped_message = ov::util::filter_lines_by_prefix(exception, get_jax_prefix());
            if (cropped_message.size()) {
                m_telemetry->send_event("error_info", cropped_message);
            }
        }
    } catch (...) {
        exception = "Unknown exception type.";
    }
    exception += "Failed to convert operation: " + context.get_op_type() + ". Reason: ";
    FRONT_END_THROW(exception);
}

void TranslateSession::encode_tensor_name(Output<Node> output,
                                          size_t tensor_idx,
                                          std::vector<std::string> additional_names) {
    if (!output.get_names().empty()) {
        OPENVINO_DEBUG("Tensor names already exist: ",
                       output.get_any_name(),
                       ". Will not be rewritten with ",
                       tensor_idx,
                       ". This is likely a mutated tensor.");
        return;
    }
    auto name = std::to_string(tensor_idx);
    std::unordered_set<std::string> names;
    names.insert(name);
    if (additional_names.size() > 0) {
        names.insert(additional_names.begin(), additional_names.end());
    }

    if (m_counter_map.count(tensor_idx)) {
        auto&& pair = m_counter_map[tensor_idx];
        auto new_name = name + '_' + std::to_string(++pair.first);
        pair.second.set_names({new_name});
        pair.second = output;
        output.set_names(names);
    } else {
        m_counter_map[tensor_idx] = {0, output};
        output.set_names(names);
    }
}

namespace {
bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it))
        ++it;
    return !s.empty() && it == s.end();
}
}  // namespace

size_t TranslateSession::decode_tensor_name(const Output<Node>& output) {
    // any_name should always return numerical value even if there is a word value
    // exist in names
    auto name = output.get_any_name();
    auto pos = name.find("_");
    if (pos != std::string::npos) {
        name = name.substr(0, pos);
    }
    // numbers after "_" will be ignored by stoll function
    FRONT_END_GENERAL_CHECK(is_number(name), "Tensor name is not a number: ", name);
    return static_cast<size_t>(std::stoll(name));
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov