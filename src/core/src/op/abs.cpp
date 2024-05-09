// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/reduce_logical_and.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/abs.hpp"
#include "compare.hpp"

namespace ov {
namespace op {
namespace abs {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::abs(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};

std::vector<bool> tensor_non_negative_or_non_interval(const ov::descriptor::Tensor& tensor) {
    const auto& lower = std::make_shared<op::v0::Parameter>(tensor.get_element_type(), tensor.get_partial_shape());
    const auto& upper = std::make_shared<op::v0::Parameter>(tensor.get_element_type(), tensor.get_partial_shape());
    const auto& gr_eq = std::make_shared<op::v1::GreaterEqual>(lower, op::v0::Constant::create(tensor.get_element_type(), {}, {0}));
    const auto& eq = std::make_shared<op::v1::Equal>(lower, upper);
    const auto& result = std::make_shared<op::v1::LogicalOr>(gr_eq, eq);
    ov::Model m(OutputVector{result}, ParameterVector{lower, upper});
    ov::TensorVector output(1);
    m.evaluate(output, {tensor.get_lower_value(), tensor.get_upper_value()});
    return op::v0::Constant(output[0]).cast_vector<bool>();
}

std::vector<bool> tensor_non_negative(const ov::descriptor::Tensor& tensor) {
    const auto& lower = std::make_shared<op::v0::Parameter>(tensor.get_element_type(), tensor.get_partial_shape());
    const auto& result = std::make_shared<op::v1::GreaterEqual>(lower, op::v0::Constant::create(tensor.get_element_type(), {}, {0}));
    ov::Model m(OutputVector{result}, ParameterVector{lower});
    ov::TensorVector output(1);
    m.evaluate(output, {tensor.get_lower_value()});
    return op::v0::Constant(output[0]).cast_vector<bool>();
}

}  // namespace abs

namespace v0 {
Abs::Abs(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Abs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Abs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Abs>(new_args.at(0));
}

bool Abs::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Abs_evaluate);

    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Abs_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      abs::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Abs::has_evaluate() const {
    OV_OP_SCOPE(v0_Abs_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool Abs::evaluate_lower(ov::TensorVector& output_values) const {
    auto check_vector = ov::op::abs::tensor_non_negative_or_non_interval(get_input_tensor(0));
    return std::all_of(check_vector.begin(), check_vector.end(), ov::cmp::Equal<bool>(true))
        && ov::default_lower_bound_evaluator(this, output_values);
}

bool Abs::evaluate_upper(ov::TensorVector& output_values) const {
    auto check_vector = ov::op::abs::tensor_non_negative_or_non_interval(get_input_tensor(0));
    return std::all_of(check_vector.begin(), check_vector.end(), ov::cmp::Equal<bool>(true))
           && ov::default_upper_bound_evaluator(this, output_values);
}

bool Abs::evaluate_symbol(ov::TensorSymbolVector& output_symbols) const {
    const auto& check_vector = ov::op::abs::tensor_non_negative(get_input_tensor(0));
    const auto& input_symbols = get_input_tensor(0).get_value_symbol();
    if (input_symbols.size() == check_vector.size()) {
        output_symbols.resize(1);
        output_symbols[0].resize(check_vector.size());
        for (size_t i = 0; i < check_vector.size(); ++i)
            output_symbols[0][i] = check_vector[i] ? input_symbols[i] : nullptr;
        return true;
    }
    return false;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
