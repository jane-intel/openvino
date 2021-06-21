// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

constexpr ngraph::NodeTypeInfo MKLDNNPlugin::FullyConnectedNode::type_info;

MKLDNNPlugin::FullyConnectedNode::FullyConnectedNode(const ngraph::Output<Node>& A,
                                                     const ngraph::Output<Node>& B,
                                                     const ngraph::PartialShape& output_shape,
                                                     const ngraph::element::Type output_type)
    : Op({A, B}), m_output_shape(output_shape), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

MKLDNNPlugin::FullyConnectedNode::FullyConnectedNode(const ngraph::Output<Node>& A,
                                                     const ngraph::Output<Node>& B,
                                                     const ngraph::Output<Node>& C,
                                                     const ngraph::PartialShape& output_shape,
                                                     const ngraph::element::Type output_type)
    : Op({A, B, C}), m_output_shape(output_shape), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::FullyConnectedNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<MKLDNNPlugin::FullyConnectedNode>(new_args.at(0), new_args.at(1), m_output_shape);
    } else if (new_args.size() == 3) {
        return std::make_shared<MKLDNNPlugin::FullyConnectedNode>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_shape);
    }

    throw ngraph::ngraph_error("Unsupported number of arguments for FullyConnected operation");
}

void MKLDNNPlugin::FullyConnectedNode::validate_and_infer_types() {
    m_output_type == ngraph::element::undefined ? input_value(0).get_element_type() : m_output_type;
    if (m_output_shape.rank().is_static() && m_output_shape.rank().get_length() > 1 &&
            m_output_shape[m_output_shape.rank().get_length() - 1].is_static())
        m_output_size = static_cast<size_t>(m_output_shape[m_output_shape.rank().get_length() - 1].get_length());

    auto data_pshape = get_input_partial_shape(0), w_pshape = get_input_partial_shape(1);
    auto data_rank = data_pshape.rank(), w_rank = w_pshape.rank();
    if (data_rank.is_static() && data_rank.get_length() >= 2 && w_rank.is_static() && w_rank.get_length() >= 1) {
        data_pshape[data_rank.get_length() - 1] = w_pshape[0];
        set_output_type(0, m_output_type, data_pshape);
    } else {
        set_output_type(0, m_output_type, m_output_shape);
    }
}

bool MKLDNNPlugin::FullyConnectedNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("out-size", m_output_size);
    return true;
}
