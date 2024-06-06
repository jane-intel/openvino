// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>

#include "openvino/core/core_visibility.hpp"

namespace ov {

class Symbol;

namespace symbol {
/// \brief If both symbols are valid, sets them as equal
void OPENVINO_API set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns true if both symbols are valid and are equal otherwise returns false
bool OPENVINO_API are_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);
/// \brief Returns a representative (the most distant parent) of an equality group of this symbol
std::shared_ptr<Symbol> OPENVINO_API ancestor_of(const std::shared_ptr<Symbol>& x);
}  // namespace symbol

std::shared_ptr<ov::Symbol> OPENVINO_API operator+(const std::shared_ptr<ov::Symbol>& lhs,
                                                   const std::shared_ptr<ov::Symbol>& rhs) noexcept;

/// \brief Class representing unique symbol for the purpose of symbolic shape inference. Equality of symbols is being
/// tracked by Disjoint-set data structure
/// \ingroup ov_model_cpp_api
class OPENVINO_API Symbol {
public:
    /// \brief Default constructs a unique symbol
    Symbol() = default;

private:
    friend std::shared_ptr<ov::Symbol> operator+(const std::shared_ptr<ov::Symbol>& lhs,
                                                 const std::shared_ptr<ov::Symbol>& rhs) noexcept;

    friend std::shared_ptr<Symbol> ov::symbol::ancestor_of(const std::shared_ptr<Symbol>& x);
    friend void ov::symbol::set_equal(const std::shared_ptr<Symbol>& lhs, const std::shared_ptr<Symbol>& rhs);

    std::shared_ptr<Symbol> m_parent = nullptr;
    // Structure to represent that A + B = C:
    //     root of symbol A would have a record of {B, C}
    //     root of symbol B would have a record of {A, C}
    //     root of symbol C would have no record
    // Synchronization (to transfer the record to new root) is only needed for set_equal function
    std::unordered_map<std::shared_ptr<ov::Symbol>, std::shared_ptr<ov::Symbol>> m_sum;
};

}  // namespace ov
