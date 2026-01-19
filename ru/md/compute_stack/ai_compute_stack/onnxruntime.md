sidebar_position: 1

# SpacemiT-ONNXRuntime

> **SpacemiT-ONNXRuntime** включает базовую библиотеку инференса [ONNXRuntime](https://github.com/microsoft/onnxruntime) и ускоряющий бэкенд SpacemiT-ExecutionProvider. Архитектура остается развязанной, поэтому способ использования почти полностью совпадает с версией ONNXRuntime от сообщества.

---

- [Быстрый старт](#быстрый-старт)
    - [Получение ресурсов](#получение-ресурсов)
    - [Инференс модели ONNXRuntime](#инференс-модели-onnxruntime)
    - [Бэкенд SpacemiT-ExecutionProvider](#бэкенд-spacemit-executionprovider)
    - [Быстрая проверка производительности модели](#быстрая-проверка-производительности-модели)
- [Описание ProviderOption](#описание-provideroption)
    - [`SPACEMIT_EP_INTRA_THREAD_NUM`](#spacemit_ep_intra_thread_num)
    - [`SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD`](#spacemit_ep_use_global_intra_thread)
    - [`SPACEMIT_EP_DUMP_SUBGRAPHS`](#spacemit_ep_dump_subgraphs)
    - [`SPACEMIT_EP_DEBUG_PROFILE`](#spacemit_ep_debug_profile)
    - [`SPACEMIT_EP_DUMP_TENSORS`](#spacemit_ep_dump_tensors)
    - [`SPACEMIT_EP_DISABLE_OP_TYPE_FILTER`](#spacemit_ep_disable_op_type_filter)
    - [`SPACEMIT_EP_DISABLE_OP_NAME_FILTER`](#spacemit_ep_disable_op_name_filter)
    - [`SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE`](#spacemit_ep_disable_float16_epilogue)
- [Описание демо](#описание-демо)
  - [onnxruntime_perf_test](#onnxruntime_perf_test)
  - [onnx_test_runner](#onnx_test_runner)
- [Описание операторов EP](#описание-операторов-ep)
- [Данные производительности моделей](#данные-производительности-моделей)
- [FAQ](#faq)

## Быстрый старт

#### Получение ресурсов
> &#x2139;&#xfe0f;Каталог `https://archive.spacemit.com/spacemit-ai/onnxruntime/` регулярно обновляется
~~~ bash
# Например, скачать версию 2.0.1
wget https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.1.tar.gz
~~~

#### Инференс модели ONNXRuntime
Подробности см. в документации сообщества [ONNXRuntime (Inference Overview)](https://onnxruntime.ai/docs/#onnx-runtime-for-inferencing) и в примерах кода [ONNXRuntime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples)

#### Бэкенд SpacemiT-ExecutionProvider

* C&C++

~~~
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "demo");
Ort::SessionOptions session_options;
std::unordered_map<std::string, std::string> provider_options;

// Ниже параметры EP, опционально
// provider_options["SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE"] = "1"; отключить приближенный epilogue
// provider_options["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1"; экспортировать скомпилированные EP подграфы в каталоге запуска с префиксом SpaceMITExecutionProvider_SpineSubgraph_
// provider_options["SPACEMIT_EP_DEBUG_PROFILE"] = "demo"; выгрузить профиль выполнения EP в JSON с заданным префиксом
SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session(env, net_param_path, session_options);

// ... далее использование совпадает с community ONNXRuntime
~~~

* Python

~~~
import onnxruntime as ort
import numpy as np
import spacemit_ort

eps = ort.get_available_providers()
net_param_path = "resnet18.q.onnx"

# определение provider_options такое же, как в C++
ep_provider_options = {}
# экспортировать скомпилированные EP подграфы в каталоге запуска с префиксом SpaceMITExecutionProvider_SpineSubgraph_
# ep_provider_options ["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1";
# выгрузить профиль выполнения EP в JSON с заданным префиксом
# ep_provider_options ["SPACEMIT_EP_DEBUG_PROFILE"] = "demo";

session = ort.InferenceSession(net_param_path,
                providers=["SpaceMITExecutionProvider"],
                provider_options=[ep_provider_options ])

input_tensor = np.ones((1, 3, 224, 224), dtype=np.float32)
outputs = session.run(None, {"data": input_tensor})
~~~

#### Быстрая проверка производительности модели
> &#x2139;&#xfe0f;Можно ориентироваться на [spacemit-demo](https://github.com/spacemit-com/spacemit-demo)

---

## Описание ProviderOption
#### `SPACEMIT_EP_INTRA_THREAD_NUM`
>+ Отдельно задает число потоков EP и не зависит от ONNXRuntime `intra_thread_num`
>+ &#x2139;&#xfe0f;Если задан только `session_options.intra_thread_num`, это эквивалентно `SPACEMIT_EP_INTRA_THREAD_NUM = intra_thread_num`, и запускается `2 * intra_thread_num` вычислительных потоков
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = "4";
~~~

#### `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD`
>+ В одном процессе используется единый пул intra-потоков для всех сессий с EP
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// "1" означает enable, любые другие значения — disable
provider_options["SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD"] = "1";

SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session0(env, net_param_path, session_options);

SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session1(env, net_param_path, session_options);

// session0 и session1 будут совместно использовать ресурсы потоков EP;
// убедитесь, что session0 и session1 не выполняют инференс одновременно
~~~

#### `SPACEMIT_EP_DUMP_SUBGRAPHS`
>+ Экспортирует подграфы, скомпилированные EP, в каталоге запуска с префиксом `SpaceMITExecutionProvider_SpineSubgraph_`
>+ &#x2139;&#xfe0f;Полезно, чтобы проверить, на сколько подграфов EP разделил модель. Чем меньше подграфов, тем лучше
>+ &#x2139;&#xfe0f;Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// "1" означает enable, любые другие значения — disable
provider_options["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1";
~~~

#### `SPACEMIT_EP_DEBUG_PROFILE`
>+ Экспортирует профиль выполнения EP в JSON с заданным префиксом. Профиль независим от профиля ONNXRuntime
>+ JSON можно открыть в Google Trace или [Perfetto](https://www.ui.perfetto.dev/) и посмотреть время выполнения каждого оператора
>+ Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// Это значение задает префикс для JSON-файлов профиля
provider_options["SPACEMIT_EP_DEBUG_PROFILE"] = "profile_";
~~~

#### `SPACEMIT_EP_DUMP_TENSORS`
>+ Выгружает промежуточные результаты EP во время выполнения в каталог с заданным именем
>+ &#x2139;&#xfe0f;Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// Это значение задает каталог для сохранения выгруженных тензоров (.npy);
// если каталога нет, EP создаст его автоматически
provider_options["SPACEMIT_EP_DUMP_TENSORS"] = "dump";
~~~

#### `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER`
>+ Запрещает EP выполнять инференс для некоторых типов операторов
>+ &#x2139;&#xfe0f;Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// Разделяйте типы операторов символом ;
provider_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv;Gemm";
~~~

#### `SPACEMIT_EP_DISABLE_OP_NAME_FILTER`
>+ Запрещает EP выполнять инференс для операторов с заданными именами
>+ &#x2139;&#xfe0f;Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// Разделяйте имена операторов символом ;
provider_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv1;Conv2";
~~~

#### `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE`
>+ Отключает FP16-эпилог, например оптимизации FP16 scaling для Conv/Gemm в режиме квантования
>+ &#x2139;&#xfe0f;Поддерживается настройка через переменные окружения
~~~ C++
std::unordered_map<std::string, std::string> provider_options;
// "1" означает отключение, другие значения — недействительны
provider_options["SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE"] = "1";
~~~

## Описание демо

### onnxruntime_perf_test
* Shell
~~~ bash
# cd spacemit-ort.riscv64.x.x.x
export LD_LIBRARY_PATH=./lib
# запуск
./bin/onnxruntime_perf_test resnet50.q.onnx -e spacemit -r 1 -x 1 -c 1 -S 1 -I
~~~

* Описание часто используемых параметров:
>+ -e: указать используемый бэкенд, например `-e spacemit`
>+ -r: число прогонов, например `-r 10`
>+ -x: число потоков
>+ -c: количество одновременно запускаемых сессий инференса
>+ -S: задать seed для воспроизводимых входных данных, по умолчанию -1
>+ -I: включить предвыделение и привязку входных тензоров
>+ -s: показать статистику
>+ -p: сформировать лог выполнения

### onnx_test_runner
onnx_test_runner используется для валидации ONNX-моделей и проверки корректности и согласованности результатов на разных бэкендах выполнения (например, SpaceMITExecutionProvider). Основные цели:
> - Проверка корректности модели: убедиться, что модель ONNX, экспортированная из обучающего фреймворка (например, PyTorch или TensorFlow), дает результаты в ONNX Runtime, совпадающие с исходным фреймворком.
> - Проверка поддержки операторов: определить, поддерживаются ли операторы модели выбранным провайдером выполнения (например, CPU, GPU, NPU).
> - Проверка кросс-платформенной согласованности: гарантировать сопоставимые результаты запуска одной и той же модели на разных аппаратных бэкендах (например, CPU, ArmNN, ACL, CUDA).
> - Вспомогательная оценка производительности: хотя основной фокус — корректность, время выполнения дает базовый ориентир по производительности (для точного бенчмаркинга используйте `onnxruntime_perf_test`).

* Shell
~~~ bash
onnx_test_runner [опции] <каталог_тестовых_данных>

# добавьте переменную окружения
export LD_LIBRARY_PATH=PATH_TO_YOUR_LIB
# запуск
./onnx_test_runner {.....}/resnet50.q-imagenet -j 1 -X 4 -e spacemit
~~~

* Структура каталога тестовых данных
~~~
mobilenetv2
├── mobilenetv2.onnx      # файл модели ONNX
└── test_data0            # каталог тестовых данных
    ├── input0.pb         # входной тензор (формат protocol buffer)
    └── output_0.pb       # ожидаемый выходной тензор
└── test_data1
    ├── input0.pb
    └── output_0.pb
~~~

* Описание часто используемых параметров
>+ -e указывает EP — это один из ключевых параметров выбора аппаратного бэкенда.
>+ -X число параллельных тестовых потоков.
>+ -c число параллельных сессий.
>+ -r число повторов теста (полезно для проверки стабильности).
>+ -v выводит более подробную информацию.

## [Описание операторов EP](./onnxruntime_ep_ops.md)
> &#x2139;&#xfe0f;Операторы, поддерживаемые EP, могут быть сгруппированы в подграфы и выполнены EP-бэкендом; неподдерживаемые операторы выполняются через CPUProvider ONNXRuntime
> &#x2139;&#xfe0f;Список поддерживаемых операторов постоянно расширяется; при наличии узких мест в моделях можно направлять запросы на поддержку

## [Данные производительности моделей](./modelzoo.md)
> Данные производительности получены при инференсе через ONNXRuntime + SpaceMITExecutionProvider с использованием onnxruntime_perf_test

## [FAQ](./onnxruntime_ep_faq.md)
> Вопросы можно задавать в [сообществе разработчиков SpacemiT](https://forum.spacemit.com/); мы постараемся ответить как можно быстрее
