# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestUnique2(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, sorted, return_inverse, return_counts):
        import torch

        class aten_unique2_return_first(torch.nn.Module):
            def __init__(self, sorted):
                super(aten_unique2_return_first, self).__init__()
                self.op = torch._unique2
                self.sorted = sorted

            def forward(self, x):
                result, inverse, _ = self.op(x, self.sorted, True, False)
                return result, inverse
            
        class aten_unique2_return_second(torch.nn.Module):
            def __init__(self, sorted):
                super(aten_unique2_return_second, self).__init__()
                self.op = torch._unique2
                self.sorted = sorted

            def forward(self, x):
                result, _, counts = self.op(x, self.sorted, False, True)
                return result, counts
            
        class aten_unique2_return_both(torch.nn.Module):
            def __init__(self, sorted):
                super(aten_unique2_return_both, self).__init__()
                self.op = torch._unique2
                self.sorted = sorted

            def forward(self, x):
                result, inverse, counts = self.op(x, self.sorted, True, True)
                return result, inverse, counts
            
        class aten_unique2_return_neither(torch.nn.Module):
            def __init__(self, sorted):
                super(aten_unique2_return_neither, self).__init__()
                self.op = torch._unique2
                self.sorted = sorted

            def forward(self, x):
                result, _, _ = self.op(x, self.sorted, False, False)
                return result

        ref_net = None
        if return_inverse and return_counts:
            model_class, op = (aten_unique2_return_both, "aten::_unique2")
        elif return_inverse:
            model_class, op = (aten_unique2_return_first, "aten::_unique2")
        elif return_counts:
            model_class, op = (aten_unique2_return_second, "aten::_unique2")
        else:
            model_class, op = (aten_unique2_return_neither, "aten::_unique2")

        return model_class(sorted), ref_net, op

    @pytest.mark.parametrize("input_shape", [
        [115], [24, 30], [5, 4, 6], [3, 7, 1, 4], [16, 3, 32, 32]
    ])
    @pytest.mark.parametrize("sorted", [False, True])
    @pytest.mark.parametrize("return_inverse", [False, True])
    @pytest.mark.parametrize("return_counts", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unique2(self, input_shape, sorted, return_inverse, return_counts, ie_device, precision, ir_version):
        self.input_tensor = np.random.randint(0, 10, size=input_shape).astype(np.int32)
        self._test(*self.create_model(sorted, return_inverse, return_counts), ie_device, precision, ir_version)
