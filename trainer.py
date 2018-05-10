from src.trainer_v0 import *
from src.predictor_v0 import *
from keras.optimizers import Adam, Nadam, SGD
import src.zoo_losses_K as zoo_losses_K

print('Import done')

if os.name == 'nt':
    GLOBAL_PATH = 'E:\Datasets\\dsbowl2018\\'
else:
    GLOBAL_PATH = '/home/nosok/datasets/dsBowl2018/'

phase = 'test'
unet_input = 224
batch_size = 24
dropout = 0.3
batch_norm = True
predict_seeds = True
use_bnds = False
predict_on_crops = False
norm_type = 'mean_std'
n_folds = 5
exclude_folds = [] #[0,1,2,3]
use_model = ZF_Seg_ResNet50_224x224
lr_sched = [[100, 0.0002, '1']]
loss = zoo_losses_K.dice_coef_and_binary_loss
metrics = zoo_losses_K.dice_coef

classes = 1
if predict_seeds:
    classes = 2

seed = 66

if os.name != 'nt':
    use_multiprocessing = True
else:
    use_multiprocessing = False

if phase == 'train':
    print('Training stage')
    trainer = Trainer(model=use_model, optimizer=Nadam,
                      batch_size=batch_size, input_size=unet_input, norm_function=norm_type,
                      loss=loss, classes=classes,
                      predict_seeds=predict_seeds,
                      lr_schedule=lr_sched,
                      n_folds=n_folds, exclude_folds=exclude_folds,
                      seed=seed,
                      use_multiprocessing=use_multiprocessing,
                      batch_norm=batch_norm,
                      dropout=dropout)
    trainer.train()

if phase == 'test':
    print('Testing phase')
    Test_dataset = BowlDataset(train_fraction=1, seed=1, phase='test', dataset_return_ids_only=True)
    test_data = Test_dataset.test_data
    predictor = Predictor(phase='test', dataset=test_data, model=use_model, model_input_size=unet_input,
                          path=GLOBAL_PATH,
                          predict_on_crops=predict_on_crops,
                          n_classes=classes, normalize=norm_type,
                          n_folds=n_folds, exclude_folds=exclude_folds,
                          nb_tta_times=2,
                          mask_threshold=0.5, seed_threshold=0.7, seed_min_size=4,
                          debug=True,
                          tf_device="/gpu:0",
                          dataset_return_ids_only=True)
    predictor.predict()

