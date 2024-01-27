# Шаблон Cookiecutter для тренировки нейросетей

Данный шаблон может использоваться для любых задач обучения нейросетей с использованием [`Torch Ligtning`](https://lightning.ai/). Для контроля данных здесь используется [`DVC`](https://dvc.org/), а для экспериментов [`ClearML`](https://clear.ml/).

## Что даёт данный шаблон?

С помощью данного шаблона возможно в короткий срок развернуть проект для обучения нейросети. Всё что необходимо это дописать некоторые блоки, которые зависят от вашей задачи и добавить данные.

## Как использовать шаблон?

Чтобы воспользоваться шаблоном необходимо скачать [`CookieCutter`](https://github.com/cookiecutter/cookiecutter), и ввести параметры для переменных, которые он будет запрашивать

```
    # Лучше использовать виртуальное окружение
    pip install cookiecutter

    cookiecutter https://github.com/matweykai/modeling_template.git
    # Вводите значения для переменных
```

Далее необходимо дописать код в файлах:
```
    src/dataset.py - нужно дописать логику получения обучающих данных и метод __len__

    src/datamodule.py - нужно реализовать методы prepare_data и setup
    
    src/metrics.py - добавляем используемые в проекте метрики

    src/lightning_module - в init добавляем инициализацию метрик, также можно исправить создание модели (сейчас они достаются из timm)
```

После этого в корне проекта запускаем `make setup`, который установит необходимые зависимости

## Краткая справка по структуре проекта

В директории `config` располагается конфигурация эксперимента, в которой описываются гиперпараметры эксперимента. При тюнинге нейросети вам будет достаточно поменять только этот файл.

`data` - хранит данные
`notebooks` - jupyter ноутбуки для EDA
`scripts` - скрипты для получения датасета и различных манипуляций, которые не связаны с экспериментом напрямую
`weights` - используется для хранения весов важных контрольных точек

Директория `src` имеет наибольшее число файлов. Тут располагается код для запуска Lightning проекта. Сюда можно добавлять свой код для решения задачи (анализ ошибок, доп аугментации, особая предобработка данных, свои модели и тд).