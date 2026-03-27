import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnetb0(num_classes, img_size=224, frozen=True, unfreeze_top_n=20, 
                         dropout=0.4, global_pooling="avg"):
    """ 
    build an EfficientNetB0 model for image classification 
    Args:
        num_classes(int): no. of output classes
        img_size(int): input image size (default: 224)
        frozen(bool): if True freeze the backbone else unfreeze top N layers
        unfreeze_top_n(int): if frozen=False, unfreeze top N layers
        dropout(float): dropout rate for the classifier head
        global_pooling(str): "avg" or "max" global pooling before the classifier head
     Returns:
        model(tf.keras.Model): EfficientNetB0 model

    """

    base_model = EfficientNetB0(include_top=False, weights="imagenet", 
                                input_shape=(img_size, img_size, 3))
    
    if frozen:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_top_n]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")
    x = base_model(inputs, training=False)
    
    if global_pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    elif global_pooling == "max":
        x = layers.GlobalMaxPooling2D(name="global_pool")(x)
    else:
        raise ValueError(f"Unknown pooling: {global_pooling}")
    
    # x = layers.Dense(512, activation="relu", name="dense_hidden")(x)
    # x = layers.BatchNormalization(name="bn_hidden")(x)     
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0")
    return model


def unfreeze_layers(model, unfreeze_top_n=20):
    """ Unfreeze top n layers of frozen EfficientNetB0 for fine-tuning 
    Args:
        model(tf.keras.Model): the EfficientNetB0 model built by build_efficientnetb0()
        unfreeze_top_n(int): number of top layers to unfreeze
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "Could not find EfficientNetB0 backbone in model. "
            "This function only works with models built by build_efficientnetb0().")
    
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_top_n]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

def count_trainable_params(model):
    """ 
    Count trainable and non-trainable parameters 
    Args:
        model(tf.keras.Model): the model to count parameters for 
    Returns:
        Tuple(int, int): number of trainable and non-trainable parameters
    """

    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    return trainable, non_trainable