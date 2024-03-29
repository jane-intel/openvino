// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

_OPENVINO_OP_REG(Abs, op::v0)
_OPENVINO_OP_REG(BatchNormInference, op::v0)
_OPENVINO_OP_REG(Ceiling, op::v0)
_OPENVINO_OP_REG(Convert, op::v0)
_OPENVINO_OP_REG(CTCGreedyDecoder, op::v0)
_OPENVINO_OP_REG(CumSum, op::v0)
_OPENVINO_OP_REG(DetectionOutput, op::v0)
_OPENVINO_OP_REG(Elu, op::v0)
_OPENVINO_OP_REG(FakeQuantize, op::v0)
_OPENVINO_OP_REG(Gelu, op::v0)
_OPENVINO_OP_REG(GRN, op::v0)
_OPENVINO_OP_REG(HardSigmoid, op::v0)
_OPENVINO_OP_REG(Interpolate, op::v0)
_OPENVINO_OP_REG(LRN, op::v0)
_OPENVINO_OP_REG(LSTMCell, op::v0)
_OPENVINO_OP_REG(LSTMSequence, op::v0)
_OPENVINO_OP_REG(MVN, op::v0)
_OPENVINO_OP_REG(NormalizeL2, op::v0)
_OPENVINO_OP_REG(PriorBox, op::v0)
_OPENVINO_OP_REG(Proposal, op::v0)
_OPENVINO_OP_REG(PSROIPooling, op::v0)
_OPENVINO_OP_REG(RegionYolo, op::v0)
_OPENVINO_OP_REG(Relu, op::v0)
_OPENVINO_OP_REG(ReorgYolo, op::v0)
_OPENVINO_OP_REG(ReverseSequence, op::v0)
_OPENVINO_OP_REG(RNNCell, op::v0)
_OPENVINO_OP_REG(Selu, op::v0)
_OPENVINO_OP_REG(Sign, op::v0)
_OPENVINO_OP_REG(SquaredDifference, op::v0)
_OPENVINO_OP_REG(TensorIterator, op::v0)
_OPENVINO_OP_REG(ROIPooling, op::v0)

_OPENVINO_OP_REG(AvgPool, op::v1)
_OPENVINO_OP_REG(BinaryConvolution, op::v1)
_OPENVINO_OP_REG(ConvertLike, op::v1)
_OPENVINO_OP_REG(Convolution, op::v1)
_OPENVINO_OP_REG(ConvolutionBackpropData, op::v1)
_OPENVINO_OP_REG(DeformablePSROIPooling, op::v1)
_OPENVINO_OP_REG(Divide, op::v1)
_OPENVINO_OP_REG(Equal, op::v1)
_OPENVINO_OP_REG(Greater, op::v1)
_OPENVINO_OP_REG(GroupConvolution, op::v1)
_OPENVINO_OP_REG(GroupConvolutionBackpropData, op::v1)
_OPENVINO_OP_REG(DeformableConvolution, op::v1)
_OPENVINO_OP_REG(LessEqual, op::v1)
_OPENVINO_OP_REG(LogicalAnd, op::v1)
_OPENVINO_OP_REG(LogicalOr, op::v1)
_OPENVINO_OP_REG(LogicalXor, op::v1)
_OPENVINO_OP_REG(LogicalNot, op::v1)
_OPENVINO_OP_REG(MaxPool, op::v1)
_OPENVINO_OP_REG(Mod, op::v1)
_OPENVINO_OP_REG(Multiply, op::v1)
_OPENVINO_OP_REG(NonMaxSuppression, op::v1)
_OPENVINO_OP_REG(OneHot, op::v1)
_OPENVINO_OP_REG(Pad, op::v1)
_OPENVINO_OP_REG(Split, op::v1)
_OPENVINO_OP_REG(Reshape, op::v1)
_OPENVINO_OP_REG(Select, op::v1)
_OPENVINO_OP_REG(GatherTree, op::v1)

_OPENVINO_OP_REG(Assign, op::v3)
_OPENVINO_OP_REG(Bucketize, op::v3)
_OPENVINO_OP_REG(EmbeddingBagOffsetsSum, op::v3)
_OPENVINO_OP_REG(EmbeddingBagPackedSum, op::v3)
_OPENVINO_OP_REG(ExtractImagePatches, op::v3)
_OPENVINO_OP_REG(EmbeddingSegmentsSum, op::v3)
_OPENVINO_OP_REG(GRUCell, op::v3)
_OPENVINO_OP_REG(NonMaxSuppression, op::v3)
_OPENVINO_OP_REG(NonZero, op::v3)
_OPENVINO_OP_REG(ReadValue, op::v3)
_OPENVINO_OP_REG(ScatterNDUpdate, op::v3)
_OPENVINO_OP_REG(ShapeOf, op::v3)

_OPENVINO_OP_REG(CTCLoss, op::v4)
_OPENVINO_OP_REG(LSTMCell, op::v4)
_OPENVINO_OP_REG(NonMaxSuppression, op::v4)
_OPENVINO_OP_REG(Proposal, op::v4)

_OPENVINO_OP_REG(BatchNormInference, op::v5)
_OPENVINO_OP_REG(GatherND, op::v5)
_OPENVINO_OP_REG(GRUSequence, op::v5)
_OPENVINO_OP_REG(LogSoftmax, op::v5)
_OPENVINO_OP_REG(LSTMSequence, op::v5)
_OPENVINO_OP_REG(Loop, op::v5)
_OPENVINO_OP_REG(LSTMSequence, op::v5)
_OPENVINO_OP_REG(NonMaxSuppression, op::v5)
_OPENVINO_OP_REG(RNNSequence, op::v5)
_OPENVINO_OP_REG(Round, op::v5)

_OPENVINO_OP_REG(Assign, op::v6)
_OPENVINO_OP_REG(CTCGreedyDecoderSeqLen, op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronDetectionOutput, op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronGenerateProposalsSingleImage, op::v6)
_OPENVINO_OP_REG(GenerateProposals, op::v9)
_OPENVINO_OP_REG(ExperimentalDetectronPriorGridGenerator, op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronROIFeatureExtractor, op::v6)
_OPENVINO_OP_REG(ExperimentalDetectronTopKROIs, op::v6)
_OPENVINO_OP_REG(GatherElements, op::v6)
_OPENVINO_OP_REG(MVN, op::v6)
_OPENVINO_OP_REG(ReadValue, op::v6)

_OPENVINO_OP_REG(DFT, op::v7)
_OPENVINO_OP_REG(Einsum, op::v7)
_OPENVINO_OP_REG(IDFT, op::v7)
_OPENVINO_OP_REG(Roll, op::v7)

_OPENVINO_OP_REG(AdaptiveAvgPool, op::v8)
_OPENVINO_OP_REG(AdaptiveMaxPool, op::v8)
_OPENVINO_OP_REG(Gather, op::v8)
_OPENVINO_OP_REG(MatrixNms, op::v8)
_OPENVINO_OP_REG(MulticlassNms, op::v8)
_OPENVINO_OP_REG(DeformableConvolution, op::v8)
_OPENVINO_OP_REG(If, op::v8)
_OPENVINO_OP_REG(GatherND, op::v8)
_OPENVINO_OP_REG(DetectionOutput, op::v8)
_OPENVINO_OP_REG(NV12toRGB, op::v8)
_OPENVINO_OP_REG(NV12toBGR, op::v8)
_OPENVINO_OP_REG(I420toRGB, op::v8)
_OPENVINO_OP_REG(I420toBGR, op::v8)

_OPENVINO_OP_REG(Sigmoid, op::v0)
_OPENVINO_OP_REG(Tanh, op::v0)
_OPENVINO_OP_REG(Exp, op::v0)
_OPENVINO_OP_REG(Log, op::v0)
_OPENVINO_OP_REG(PriorBox, op::v8)
_OPENVINO_OP_REG(PRelu, op::v0)

_OPENVINO_OP_REG(GridSample, op::v9)
_OPENVINO_OP_REG(RDFT, op::v9)
_OPENVINO_OP_REG(NonMaxSuppression, op::v9)
_OPENVINO_OP_REG(IRDFT, op::v9)

_OPENVINO_OP_REG(ROIAlign, op::v9)
_OPENVINO_OP_REG(SoftSign, op::v9)
_OPENVINO_OP_REG(MulticlassNms, op::v9)

_OPENVINO_OP_REG(IsFinite, op::v10)
_OPENVINO_OP_REG(IsInf, op::v10)
_OPENVINO_OP_REG(IsNaN, op::v10)
_OPENVINO_OP_REG(Unique, op::v10)

_OPENVINO_OP_REG(Interpolate, op::v11)

_OPENVINO_OP_REG(GroupNormalization, ov::op::v12)

_OPENVINO_OP_REG(BitwiseAnd, ov::op::v13)
_OPENVINO_OP_REG(BitwiseNot, ov::op::v13)
_OPENVINO_OP_REG(BitwiseOr, ov::op::v13)
_OPENVINO_OP_REG(BitwiseXor, ov::op::v13)
_OPENVINO_OP_REG(NMSRotated, ov::op::v13)
_OPENVINO_OP_REG(Multinomial, ov::op::v13)

_OPENVINO_OP_REG(Inverse, ov::op::v14)
_OPENVINO_OP_REG(AvgPool, ov::op::v14)
_OPENVINO_OP_REG(MaxPool, ov::op::v14)

_OPENVINO_OP_REG(AUGRUCell, ov::op::internal)
_OPENVINO_OP_REG(AUGRUSequence, ov::op::internal)
