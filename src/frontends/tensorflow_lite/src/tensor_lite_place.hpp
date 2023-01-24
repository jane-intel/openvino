// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"
#include "place.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

struct Quantization {
    std::vector<float> scale;
    std::vector<int64_t> zero_point;
    int64_t axis{};
    bool no_quantization = true;
};

class TensorLitePlace : public ov::frontend::tensorflow::TensorPlace {
public:
    TensorLitePlace(const ov::frontend::InputModel& input_model,
                    const ov::PartialShape& pshape,
                    ov::element::Type type,
                    const std::vector<std::string>& names,
                    std::shared_ptr<ov::frontend::tensorflow_lite::Quantization> quantization,
                    int64_t input_idx,
                    int64_t output_idx,
                    const void* data)
        : ov::frontend::tensorflow::TensorPlace(input_model, pshape, type, names),
          m_quantization(std::move(quantization)),
          m_input_idx(input_idx),
          m_output_idx(output_idx),
          m_data(data){};

    std::shared_ptr<Quantization> get_quantization() const;
    void disable_quantization();

    void translate(ov::Output<ov::Node>& output, bool convert_tensor_attrs_to_nodes = false);

    bool is_input() const override {
        return m_input_idx >= 0;
    }
    size_t get_input_index() const {
        FRONT_END_GENERAL_CHECK(is_input(), "This is not input TensorPlace. Can not deliver input index");
        return static_cast<size_t>(m_input_idx);
    }
    bool is_output() const override {
        return m_output_idx >= 0;
    }
    size_t get_output_index() const {
        FRONT_END_GENERAL_CHECK(is_output(), "This is not output TensorPlace. Can not deliver output index");
        return static_cast<size_t>(m_output_idx);
    }
    void set_input_index(const int64_t& idx) {
        m_input_idx = idx;
    }
    void set_output_index(const int64_t& idx) {
        m_output_idx = idx;
    }

    const void* get_data() const {
        return m_data;
    }

protected:
    std::shared_ptr<ov::frontend::tensorflow_lite::Quantization> m_quantization;
    int64_t m_input_idx, m_output_idx;
    const void* m_data;
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
