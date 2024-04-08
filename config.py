import argparse
from train import *

if __name__ == '__main__':
    data_folder = "./resized_images_224"
    #if pretrained=True, load imagenet weights. else load chestXnet
    
    # choose the wanted run and uncomment it to train the model:


    ###finding best LR
    # #MAE freeze (best result: lr = 3e-3)
    # training(title="LR_AND_EPOCH_TEST_freeze_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=1e-2, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=1e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=1e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # #MAE unfreeze (best result: lr = 3e-4)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=1e-2, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=1e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=1e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)

    #classification 5 classes freeze (best result: lr = 3e-3)
    # training(title="LR_AND_EPOCH_TEST_freeze_5_classes", block_to_unfreeze=0, pretrained=False, learning_rate=1e-2, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_5_classes", block_to_unfreeze=0, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_5_classes", block_to_unfreeze=0, pretrained=False, learning_rate=1e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_5_classes", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_freeze_5_classes", block_to_unfreeze=0, pretrained=False, learning_rate=1e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # #classification 5 classes unfreeze (best result: lr = 1e-3)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_5_classes", block_to_unfreeze=3, pretrained=False, learning_rate=1e-2, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_5_classes", block_to_unfreeze=3, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_5_classes", block_to_unfreeze=3, pretrained=False, learning_rate=1e-3, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_5_classes", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)
    # training(title="LR_AND_EPOCH_TEST_unfreeze_5_classes", block_to_unfreeze=3, pretrained=False, learning_rate=1e-4, batch_size=64, num_epochs=50, data_folder=data_folder, num_classes=5, data_frac=1, dino=False)


    ###fine tuning vs feature extraction:
    # FEATURE EXTRACTION:
    # training(title="feature_extraction_MAE", block_to_unfreeze=0, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # FINE TUNING:
    # training(title="fine_tuning_1_block_MAE", block_to_unfreeze=1, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="fine_tuning_2_block_MAE", block_to_unfreeze=2, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="fine_tuning_3_block_MAE", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)


    ###different training size:
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.01, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.05, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.1, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.2, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.4, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.6, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.8, dino=False)
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    ## give low data_frac more epochs for comperable steps:
    # training(title="data_size_test", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=2000, data_folder=data_folder, num_classes=1, data_frac=0.01, dino=False)
    ## try low data with freeze t osee if the only thing we realy need to learn is the classifier's weights:
    # training(title="data_size_test_with_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=300, data_folder=data_folder, num_classes=1, data_frac=0.01, dino=False)
    # training(title="data_size_test_with_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=60, data_folder=data_folder, num_classes=1, data_frac=0.05, dino=False)
    # training(title="data_size_test_with_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.1, dino=False)
    # training(title="data_size_test_with_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=0.2, dino=False)

    ### imagenet vs chesXnet vs dino
    #freeze
    # training(title="pretrain_test_imagenet_freeze", block_to_unfreeze=0, pretrained=True, learning_rate=3e-3, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="pretrain_test_chesXnet_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="pretrain_test_dino_freeze", block_to_unfreeze=0, pretrained=False, learning_rate=3e-3, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=True)
    # #unfreeze
    # training(title="pretrain_test_imagenet_unfreeze", block_to_unfreeze=3, pretrained=True, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="pretrain_test_chesXnet_unfreeze", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=False)
    # training(title="pretrain_test_dino_unfreeze", block_to_unfreeze=3, pretrained=False, learning_rate=3e-4, batch_size=64, num_epochs=40, data_folder=data_folder, num_classes=1, data_frac=1, dino=True)



