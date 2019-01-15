# DSBowl2018
Final pipeline for DSB2018 

Task descriprion: instance segmentation of nuclei 
https://www.kaggle.com/c/data-science-bowl-2018

## Описание решения:
Используется подход с использованием encoder-decoder архитектуры (Unet с различными экнодерами) и watershed transform для разделения инстансов
<br>

Пайплайн:

1. Препроцессинг (src.dataset_v0.py):
    1. Препроцессинг маски:
        1. Граница маски как разница cv2.dilate() и cv2.erode() исходной маски
        2. Ядро маски как cv2.erode() исходной маски
        3. "Энергия" для каждой nuclei как величина прямо пропорциональную удалению от центра клетки (см. функцию get_distance() в src.dataset_v0)
    2. Препроцессинг изображения: нормализация (div 255 / mean std / imagenet / CLAHE)


2. Аугментации (src.dataset_v0.py, src.augs.py):
    1. random shift, scale, rotate, flip
    2. random crop: кропаются только участки с площадью маски > threshold
    3. random put: если изображение меньше входа сети, то генерируется canvas и изображение помещается в случайное место на нем
    4. random contrast, brightness


3. Сеть: resnet50-unet с двумя (тремя) выходами: бинарная маска, центр каждой клетки (граница клетки)


4. Тренировка (src.trainer_v0.py)
    1. 5 фолдов, без стратификации
    2. loss: dice_bce / weighted dice_bce, веса - из п. 1.1.3 препроцессинга
    3. lr schedule с понижением lr каждые 50 эпох
    4. Cyclic lr
    5. Для тренировки использовались доп. данные (но не для валидации)


5. Инференс (src.predictor_v0.py):
    1. Большие изображения (больше входа сети): crop+overlap со stride=16px. На перекрываемых областях - усреднение
    2. TTA: rotate 90-180-270
    3. оптимизация threshold по train+valid выборке, усреднение по фолдам
    4. watershed transform

## How to run?
Clone this repo
Modify trainer.py:
- Change GLOBAL_PATH (dataset folder)
- Modify training params:
-- phases = ['train', 'test'] 
-- unet_input = 224
-- batch_size = 24
-- dropout = 0.3
-- batch_norm = True
-- predict_seeds = True - seed are center of nuclei (cv2.erode())
-- use_bnds = False
-- predict_on_crops = False
-- norm_type = 'mean_std'
-- n_folds = 5
-- exclude_folds = []
-- use_model = ZF_Seg_ResNet50_224x224
-- lr_sched = [[100, 0.0002, '1']]

## How to improve?
- Implement FPN - it should give ~10% boost
- Try focal loss / bce+lovasz loss
- Pseudolabeleing
