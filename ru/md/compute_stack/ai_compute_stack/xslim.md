sidebar_position: 4

# XSlim
> **XSlim** — это инструмент PTQ-квантования от **SpacemiT**. Он интегрирует настроенные под чип стратегии квантования и предоставляет единый интерфейс через JSON-конфиг для квантования моделей. Проект полностью открыт на GitHub: [github-xslim](https://github.com/spacemit-com/xslim)


---

- [Быстрый старт](#быстрый-старт)
- [Настройка параметров квантования](#настройка-параметров-квантования)
- [Тонкая настройка точности квантования](#тонкая-настройка-точности-квантования)
- [Журнал изменений](#журнал-изменений)

## Быстрый старт
- Установка
~~~
pip install xslim
~~~

- Python
~~~ python
import xslim

demo_json = dict()
# заполните demo_json требуемыми полями

demo_json_path = "./demo_json.json"
# Использовать словарь
xslim.quantize_onnx_model(demo_json)
# Использовать JSON-файл
xslim.quantize_onnx_model(demo_json_path)

# При вызове API можно передать путь к модели или ModelProto
# xslim.quantize_onnx_model("resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx")

# xslim.quantize_onnx_model(
#    "resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx", "resnet18_output.onnx"
# )

# import onnx
# onnx_model = onnx.load("/home/share/modelzoo/classification/resnet18/resnet18.onnx")
# quantized_onnx_model = xslim.quantize_onnx_model("resnet18.json", onnx_model)
~~~

- Командная строка
~~~ bash
python -m xslim --config ./demo_json.json
# Указать пути входной и выходной моделей
python -m xslim -c ./demo_json.json -i demo.onnx -o demo.q.onnx
# Использовать динамическое квантование, JSON не требуется
python -m xslim -i demo.onnx -o demo.q.onnx --dynq
# Преобразовать в FP16, JSON не требуется
python -m xslim -i demo.onnx -o demo.q.onnx --fp16
# Только упрощение модели без квантования, JSON не требуется
python -m xslim -i demo.onnx -o demo.q.onnx
~~~

---

## Настройка параметров квантования
- Пример JSON-конфига
~~~
{
    "model_parameters" : {
        "onnx_model": "", "путь к ONNX-модели"
        "output_prefix": "", "можно опустить, префикс имени выходной квантованной модели"
        "working_dir": "" "можно опустить, каталог вывода и файлов, создаваемых при квантовании"
        "skip_onnxsim": false "пропустить упрощение модели через onnxsim, по умолчанию false"
    },
    "calibration_parameters" : {
        "calibration_step": 100, "можно опустить, максимум калибровочных файлов, по умолчанию 100"
        "calibration_device": "cpu", "можно опустить, по умолчанию cuda, автоопределение, иначе cpu"
        "calibration_type": "default",  "можно опустить, по умолчанию default; варианты: kl, minmax, percentile, mse"
        "input_parametres": [
            {
                "input_name": "data", "можно опустить, читается из модели",
                "input_shape": [1, 3, 224, 224], "можно опустить, форма входа читается из модели"
                "dtype": "float32", "можно опустить, тип входных данных читается из модели"
                "file_type": "img", "можно опустить, по умолчанию img; варианты: img, npy, raw"
                "color_format": "bgr", "можно опустить, по умолчанию bgr"
                "mean_value": [103.94, 116.78, 123.68], "можно опустить, по умолчанию пусто"
                "std_value": [57, 57, 57], "можно опустить, по умолчанию пусто"
                "preprocess_file": "", "скрипт py для кастомной предобработки,
                                        есть предустановленные поля PT_IMAGENET, IMAGENET"
                "data_list_path": "" "обязательное поле, путь к списку калибровочных данных"
            },
            {
                "input_name": "data1",
                "input_shape": [1, 3, 224, 224],
                "dtype": "float32",
                "file_type": "img",
                "mean_value": [103.94, 116.78, 123.68],
                "std_value": [57, 57, 57],
                "preprocess_file": "",
                "data_list_path": ""
            }
        ]
    },
    "Ниже все можно опустить"
    "quantization_parameters": {
        "analysis_enable": true, "включить анализ после квантования, по умолчанию true"
        "precision_level": 0
        "finetune_level": 1 "по умолчанию 1; варианты: 0, 1, 2, 3"
        "max_percentile": 0.9999 "указать порог percentile-квантования"
        "custom_setting": [
            "все квантованные операторы, ограниченные входными и выходными ребрами; на входе можно не указывать константы"
            {
                "input_names": ["aaa", "bbb"],
                "output_names": ["ccc"],
                "max_percentile": 0.999,
                "precision_level": 2,
                "calibration_type": "default"
            }
        ],
        "truncate_var_names": ["/Concat_5_output_0", "/Transpose_6_output_0"] "усечение модели"
    }
}
~~~

- Поля, которые можно опустить

| Поле | По умолчанию | Допустимые значения | Примечания |
| --- | --- | --- | --- |
| output_prefix | имя файла ONNX без суффикса, выход оканчивается на .q.onnx | / |  |
| working_dir | каталог, где находится onnx_model | / |  |
| calibration_step | 100 |  | рекомендуемый диапазон 100-1000 |
| calibration_device | cuda | cuda, cpu | определяется автоматически |
| calibration_type | default | default, kl, minmax, percentile, mse | рекомендуется начать с default, затем попробовать percentile или minmax |
| input_name | читается из модели ONNX |  |  |
| input_shape | читается из модели ONNX |  | форма должна быть целочисленной; символический batch поддерживается и по умолчанию равен 1 |
| dtype | читается из модели ONNX<br> | float32, int8, uint8, int16<br> | - сейчас поддерживается только float32 |
| file_type | img | img, npy, raw | - для raw поддерживаются только данные, совпадающие с dtype (по умолчанию float32) |
| preprocess_file | None | PT_IMAGENET, IMAGENET | встроены два стандартных варианта предобработки ImageNet |
| finetune_level | 1 | 0, 1, 2, 3 | - 0: без агрессивной подстройки параметров<br>- 1: возможна статическая настройка параметров квантования<br>- 2+: настройка параметров квантования по блокам на основе потерь от квантования |
| precision_level<br> | 0 | 0, 1, 2, 3, 4<br> | - 0: полное int8-квантование, даже при тюнинге остается int8<br>- 1-2: частичное int8-квантование (подходит для большинства Transformer-моделей)<br>- 3: динамическое квантование<br>- 4: FP16 |
| max_percentile | 0.9999 |  | диапазон усечения для percentile-квантования, минимум 0.99 |
| custom_setting | None |  |  |
| truncate_var_names | [] |  | граф делится по именам тензоров; результат деления проверяется, иначе ошибка |

- Правила для файла списка калибровочных данных

Каждая строка в img_list.txt — путь к одному файлу калибровочных данных. Можно указывать относительные пути (относительно расположения img_list.txt) или абсолютные. Для моделей с несколькими входами убедитесь, что порядок файлов в списках совпадает.
~~~
QuantZoo/Data/Imagenet/Calib/n01440764/ILSVRC2012_val_00002138.JPEG
QuantZoo/Data/Imagenet/Calib/n01443537/ILSVRC2012_val_00000994.JPEG
QuantZoo/Data/Imagenet/Calib/n01484850/ILSVRC2012_val_00014467.JPEG
QuantZoo/Data/Imagenet/Calib/n01491361/ILSVRC2012_val_00003204.JPEG
QuantZoo/Data/Imagenet/Calib/n01494475/ILSVRC2012_val_00015545.JPEG
QuantZoo/Data/Imagenet/Calib/n01496331/ILSVRC2012_val_00008640.JPEG
~~~

- Правила для preprocess_file

Например, если есть скрипт custom_preprocess.py, то в конфиге preprocess_file нужно указать `custom_preprocess.py:preprocess_impl`, чтобы сослаться на конкретную функцию. Для моделей с несколькими входами можно повторно использовать предобработку, если логика одинакова.
~~~ python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Прочитать path_list, выполнить предобработку согласно input_parametr
    и вернуть torch.Tensor.

    Args:
        path_list (Sequence[str]): Список файлов для одного калибровочного batch
        input_parametr (dict): Соответствует calibration_parameters.input_parametres[idx]

    Returns:
        torch.Tensor: Калибровочные данные одного batch
    """
    batch_list = []
    mean_value = input_parametr["mean_value"]
    std_value = input_parametr["std_value"]
    input_shape = input_parametr["input_shape"]
    for file_path in path_list:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
~~~

## Тонкая настройка точности квантования
> Будет добавлено

## Журнал изменений
Подробности см. в [github-xslim-releases](https://github.com/spacemit-com/xslim/releases)
