import os


def is_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def get_paths():
    if is_kaggle():
        train_test_dataset_path = '/kaggle/input/deep-learning-spring-2025-project-1/cifar-10-python'
        inference_dataset_path = '/kaggle/input/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl'
        model_history_path = '/kaggle/working/model_history'
        model_predictions_path = '/kaggle/working/predictions_history'
        model_performance_path = '/kaggle/working/performance_history'
    else:
        train_test_dataset_path = 'data'
        inference_dataset_path = 'testset/cifar_test_nolabel.pkl'
        model_history_path = 'checkpoints'
        model_predictions_path = 'predictions'
        model_performance_path = 'performance'

    return train_test_dataset_path, inference_dataset_path, model_history_path, model_predictions_path, model_performance_path