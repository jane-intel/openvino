// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/visualize_tree.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov;
using namespace std;


TEST(visualize_tree, testing) {
    auto param_1 = make_shared<op::v0::Parameter>(element::dynamic, Shape{1, 2, 3});
    param_1->set_friendly_name("param_1");

    auto relu = make_shared<op::v0::Relu>(param_1);
    relu->set_friendly_name("scope/relu");

    auto constant = make_shared<op::v0::Constant>(element::f32, Shape{1, 1, 1}, 1);
    constant->set_friendly_name("constant");

    auto sqrt = make_shared<op::v0::Sqrt>(constant);
    sqrt->set_friendly_name("scope/sqrt");

    auto mul = make_shared<op::v1::Multiply>(relu, sqrt);
    mul->set_friendly_name("scope/mul");

    auto output = make_shared<op::v0::Relu>(mul);

    auto m = make_shared<Model>(output, ParameterVector{param_1});

    pass::VisualizeTree("test.svg").run_on_model(m);
}
