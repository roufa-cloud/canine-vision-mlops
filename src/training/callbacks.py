import os
import tensorflow as tf


def create_callbacks(checkpoint_path, monitor="val_accuracy", mode="max", early_stop_patience=6, 
                     reduce_lr_patience=3, reduce_lr_factor=0.3, save_best_only=True, verbose=1):
    """ 
    Create a standard callback for training 
    Args:
        checkpoint_path(str): path to save the model checkpoint
        monitor(str): metrics to monitor 
        mode(str): "min" or "max" for monitoring metrics
        early_stop_patience(int): no. of epochs with no improvement after this epochs
        reduce_lr_patience(int): reduce LR after this epochs with no improvement
        reduce_lr_factor(float): factor to reduce LR by
        save_best_only(bool): whether to save only the best model checkpoint
        verbose(int): verbosity model (0=silent, 1=progress messages, 2=one line per epoch)
    Returns:
        list: list of callbacks
    """

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 1- EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stop_patience,
        mode=mode,
        verbose=verbose,
        restore_best_weights=True)

    # 2- ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        mode=mode,
        min_lr=1e-7,
        verbose=verbose)

    # 3- ModelCheckpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        verbose=verbose)

    return [early_stop, reduce_lr, checkpoint]


def create_tensorboard_callback(log_dir):
    """ create a TensorBoard callback for logging training metrics 
    Args:
        log_dir(str): directory to save the TensorBoard logs
    Returns:
        tf.keras.callbacks.TensorBoard: TensorBoard callback
    """

    os.makedirs(log_dir, exist_ok=True)
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq="epoch")
