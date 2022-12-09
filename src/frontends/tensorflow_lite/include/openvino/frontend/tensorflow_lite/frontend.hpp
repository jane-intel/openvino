// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>

#include "openvino/core/any.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
class TENSORFLOW_LITE_API FrontEnd : public ov::frontend::tensorflow::FrontEnd {
public:
    FrontEnd();
    /// \brief Gets name of this FrontEnd. Can be used by clients
    std::string get_name() const override {
        return "tflite";
    }

protected:
    /// \brief Check if FrontEndTensorflowLite can recognize model from given parts
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    ov::frontend::InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
};
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
