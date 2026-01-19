sidebar_position: 2

# Ускоренные операторы ONNXRuntime EP

>+ В этом разделе перечислены ускоренные операторы, поддерживаемые SpaceMITExecutionProvider, и их спецификации
>+ [Справочник ONNX-OP](https://onnx.ai/onnx/operators/index.html)
>+ [Справочник ONNX-Contrib-OP](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md)

- [Dense](#dense)
  - [**Conv**](#conv)
  - [**ConvTranspose**](#convtranspose)
  - [**Gemm**](#gemm)
  - [**MatMul**](#matmul)
- [QDQ](#qdq)
  - [**DynamicQuantizeLinear**](#dynamicquantizelinear)
  - [**QuantizeLinear**](#quantizelinear)
  - [**DequantizeLinear**](#dequantizelinear)
- [Pool](#pool)
  - [**AveragePool**](#averagepool)
  - [**GlobalAveragePool**](#globalaveragepool)
  - [**MaxPool**](#maxpool)
  - [**GlobalMaxPool**](#globalmaxpool)
- [Reduce](#reduce)
  - [**ReduceMean**](#reducemean)
  - [**ReduceMax**](#reducemax)
- [Math](#math)
  - [**Add**](#add)
  - [**Sub**](#sub)
  - [**Mul**](#mul)
  - [**Div**](#div)
  - [**Pow**](#pow)
  - [**Sqrt**](#sqrt)
  - [**Abs**](#abs)
  - [**Reciprocal**](#reciprocal)
- [Activation](#activation)
  - [**Sigmoid**](#sigmoid)
  - [**Swish**](#swish)
  - [**HardSigmoid**](#hardsigmoid)
  - [**HardSwish**](#hardswish)
  - [**Tanh**](#tanh)
  - [**LeakyRelu**](#leakyrelu)
  - [**Clip**](#clip)
  - [**Relu**](#relu)
  - [**Elu**](#elu)
  - [**Gelu**](#gelu)
  - [**Erf**](#erf)
  - [**Softmax**](#softmax)
- [Tensor](#tensor)
  - [**Cast**](#cast)
  - [**Concat**](#concat)
  - [**Split**](#split)
  - [**Transpose**](#transpose)
  - [**Unsqueeze**](#unsqueeze)
  - [**Squeeze**](#squeeze)
  - [**Reshape**](#reshape)
  - [**Flatten**](#flatten)
  - [**Gather**](#gather)
  - [**Resize**](#resize)
- [Norm](#norm)
  - [**LayerNormalization**](#layernormalization)
  - [**BatchNormalization**](#batchnormalization)

## Dense
### **Conv**
>+ Домен: ai.onnx
>+ Opset: 11
>+ Атрибуты: W должен быть константой
>+ Тип: T: tensor(float) | tensor(float16)
>+ Примечания: поддерживается формат квантования QDQ; W — симметричный per-channel, X — асимметричный per-tensor

### **ConvTranspose**
>+ Домен: ai.onnx
>+ Opset: 11
>+ Атрибуты: W должен быть константой
>+ Тип: T: tensor(float) | tensor(float16)
>+ Примечания: поддерживается формат квантования QDQ; W — симметричный per-channel, X — асимметричный per-tensor

### **Gemm**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты: transA == 0
>+ Тип: T: tensor(float) | tensor(float16)
>+ Примечания: поддерживается формат квантования QDQ; A — асимметричный per-tensor, B — симметричный per-channel; если B не является константой, он должен быть асимметричным per-tensor

### **MatMul**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)
>+ Примечания: поддерживается формат квантования QDQ; A — асимметричный per-tensor, B — симметричный per-channel; если B не является константой, он должен быть асимметричным per-tensor

## QDQ
### **DynamicQuantizeLinear**
>+ Домен: ai.onnx
>+ Opset: 11
>+ Атрибуты: поддерживается только per-tensor
>+ Тип: T1: tensor(float)
>+ Тип: T2: tensor(int8)

### **QuantizeLinear**
>+ Домен: ai.onnx
>+ Opset: 19
>+ Атрибуты: поддерживается per-tensor и per-channel
>+ Тип: T1: tensor(float)
>+ Тип: T2: tensor(int8) | tensor(int16)

### **DequantizeLinear**
>+ Домен: ai.onnx
>+ Opset: 19
>+ Атрибуты: поддерживается per-tensor и per-channel
>+ Тип: T1: tensor(int8) | tensor(int16) | tensor(int32)
>+ Тип: T2: tensor(float)

## Pool
### **AveragePool**
>+ Домен: ai.onnx
>+ Opset: 22
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **GlobalAveragePool**
>+ Домен: ai.onnx
>+ Opset: 1
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **MaxPool**
>+ Домен: ai.onnx
>+ Opset: 12
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(int8) | tensor(float16)

### **GlobalMaxPool**
>+ Домен: ai.onnx
>+ Opset: 1
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(int8) | tensor(float16)

## Reduce
### **ReduceMean**
>+ Домен: ai.onnx
>+ Opset: 18
>+ Атрибуты: axes должен быть непрерывным постоянным диапазоном, например [1,2]
>+ Тип: T: tensor(float) | tensor(float16)

### **ReduceMax**
>+ Домен: ai.onnx
>+ Opset: 20
>+ Атрибуты: axes должен быть непрерывным постоянным диапазоном, например [1,2]
>+ Тип: T: tensor(float) | tensor(int8) | tensor(float16)

## Math
### **Add**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Sub**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Mul**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Div**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Pow**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Sqrt**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Abs**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Reciprocal**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

## Activation
### **Sigmoid**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Swish**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **HardSigmoid**
>+ Домен: ai.onnx
>+ Opset: 22
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **HardSwish**
>+ Домен: ai.onnx
>+ Opset: 22
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Tanh**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **LeakyRelu**
>+ Домен: ai.onnx
>+ Opset: 16
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Clip**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Relu**
>+ Домен: ai.onnx
>+ Opset: 14
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Elu**
>+ Домен: ai.onnx
>+ Opset: 22
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Gelu**
>+ Домен: ai.onnx
>+ Opset: 20
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Erf**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

### **Softmax**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: T: tensor(float) | tensor(float16)

## Tensor
### **Cast**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Concat**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: All

### **Split**
>+ Домен: ai.onnx
>+ Opset: 18
>+ Атрибуты:
>+ Тип: All

### **Transpose**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Unsqueeze**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Squeeze**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Reshape**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Flatten**
>+ Домен: ai.onnx
>+ Opset: 24
>+ Атрибуты:
>+ Тип: All

### **Gather**
>+ Домен: ai.onnx
>+ Opset: 13
>+ Атрибуты:
>+ Тип: All

### **Resize**
>+ Домен: ai.onnx
>+ Opset: 19
>+ Атрибуты:
>+ Тип: All

## Norm
### **LayerNormalization**
>+ Домен: ai.onnx
>+ Opset: 17
>+ Атрибуты: Scale и B должны быть константами
>+ Тип: T: tensor(float) | tensor(float16)

### **BatchNormalization**
>+ Домен: ai.onnx
>+ Opset: 15
>+ Атрибуты: Scale, B, input_mean и input_var должны быть константами
>+ Тип: T: tensor(float) | tensor(float16)
