import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0


def build_fusion_model(num_classes, img_size=224, freeze_backbones=True, dense_units=256, dropout=0.5):
    """ Build a feature fusion model combining ResNet50 and EfficientNetB0  
    Args:
        num_classes(int): no. of output classes
        img_size(int): input image size (default: 224)
        frozen(bool): if True freeze the backbone 
        dense_units(int): no. of units in the fusion dense layer
        dropout(float): dropout rate for the fusion head
     Returns:
        model(tf.keras.Model): the feature fusion model
     """
    # Backbone 1: ResNet50
    resnet_base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3))
    resnet_base.trainable = not freeze_backbones
    input_resnet = layers.Input(shape=(img_size, img_size, 3), name="input_resnet50") 
    x_resnet = resnet_base(input_resnet, training=False)
    x_resnet = layers.GlobalAveragePooling2D(name="resnet_pool")(x_resnet)
    x_resnet = layers.BatchNormalization(name="resnet_bn")(x_resnet)
    x_resnet = layers.Dense(1024, activation="relu", name="resnet_dense")(x_resnet)

    # Backbone 2: EfficientNetB0
    effnet_base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3))
    effnet_base.trainable = not freeze_backbones
    input_effnet = layers.Input(shape=(img_size, img_size, 3), name="input_efficientnetb0") 
    x_effnet = effnet_base(input_effnet, training=False)
    x_effnet = layers.GlobalAveragePooling2D(name="effnet_pool")(x_effnet)
    x_effnet = layers.BatchNormalization(name="effnet_bn")(x_effnet)
    x_effnet = layers.Dense(1024, activation="relu", name="effnet_dense")(x_effnet)

    # Fusion head
    combined = layers.Concatenate(name="fusion_concat")([x_resnet, x_effnet])
    x = layers.Dense(dense_units, activation="relu", name="fusion_dense")(combined)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(dropout, name="fusion_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(
        inputs={"input_resnet50": input_resnet, "input_efficientnetb0": input_effnet}, 
        outputs=outputs,
        name="FeatureFusion_ResNet50_EfficientNetB0")

    return model


def embedding_dimensions(model):
    """ 
    Extract the embedding dimensions before fusion  
    Args:
        model(tf.keras.Model): the fusion model built by build_fusion_model()
    Returns:
        Dict[str, int]: embedding dimensions for each backbone
    """
    resnet_dense = model.get_layer("resnet_dense")
    effnet_dense = model.get_layer("effnet_dense")
    return {
        "resnet50": resnet_dense.output.shape[-1],
        "efficientnetb0": effnet_dense.output.shape[-1]}