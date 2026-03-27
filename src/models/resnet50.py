import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


def build_resnet50(num_classes, img_size=224, frozen=True, unfreeze_top_n=10, 
                   dropout=0.5, global_pooling="avg"):
    """ 
    Build a ResNet50 model for image classification
    Args:
        num_classes(int): no. of output classes
        img_size(int): input image size (default: 224)
        frozen(bool): if True freeze the backbone else unfreeze top N layers
        unfreeze_top_n(int): if frozen=False, unfreeze top N layers
        dropout(float): dropout rate for the classifier head
        global_pooling(str): "avg" or "max" global pooling before the classifier head
     Returns:
        model(tf.keras.Model): the ResNet50 model
     """

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
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
    x = base_model(inputs)
    if global_pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    elif global_pooling == "max":
        x = layers.GlobalMaxPooling2D(name="global_pool")(x)
    else:
        raise ValueError(f"Unknown pooling: {global_pooling}. Use 'avg' or 'max'.")
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50")

    return model


def unfreeze_layers(model, unfreeze_top_n=10):
    """ Unfreeze top n layers of frozen ResNet50 for fine-tuning 
    Args:
        model(tf.keras.Model): the ResNet50 model built by build_resnet50()
        unfreeze_top_n(int): number of top layers to unfreeze
    """

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == "resnet50":
            base_model = layer
            break

    if base_model is None:
        raise ValueError(
            "Could not find ResNet50 backbone in model. "
            "This function only works with models built by build_resnet50()."
        )

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