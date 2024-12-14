import tensorflow as tf

DenseNet121 = tf.keras.applications.DenseNet121
DenseNet169 = tf.keras.applications.DenseNet169
DenseNet201 = tf.keras.applications.DenseNet201
EfficientNetB0 = tf.keras.applications.EfficientNetB0
EfficientNetB1 = tf.keras.applications.EfficientNetB1
EfficientNetB2 = tf.keras.applications.EfficientNetB2
EfficientNetB3 = tf.keras.applications.EfficientNetB3
EfficientNetB4 = tf.keras.applications.EfficientNetB4
EfficientNetB5 = tf.keras.applications.EfficientNetB5
EfficientNetB6 = tf.keras.applications.EfficientNetB6
EfficientNetB7 = tf.keras.applications.EfficientNetB7
InceptionV3 = tf.keras.applications.InceptionV3
MobileNet = tf.keras.applications.MobileNet
MobileNetV2 = tf.keras.applications.MobileNetV2
NASNetLarge = tf.keras.applications.NASNetLarge
NASNetMobile = tf.keras.applications.NASNetMobile
ResNet50 = tf.keras.applications.ResNet50
ResNet50V2 = tf.keras.applications.ResNet50V2
ResNet101 = tf.keras.applications.ResNet101
ResNet101V2 = tf.keras.applications.ResNet101V2
ResNet152 = tf.keras.applications.ResNet152
ResNet152V2 = tf.keras.applications.ResNet152V2
VGG16 = tf.keras.applications.VGG16
VGG19 = tf.keras.applications.VGG19
Xception = tf.keras.applications.Xception


class KerasCatalogService:
    MODELS_DICT = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "EfficientNetB0": EfficientNetB0,
        "EfficientNetB1": EfficientNetB1,
        "EfficientNetB2": EfficientNetB2,
        "EfficientNetB3": EfficientNetB3,
        "EfficientNetB4": EfficientNetB4,
        "EfficientNetB5": EfficientNetB5,
        "EfficientNetB6": EfficientNetB6,
        "EfficientNetB7": EfficientNetB7,
        "InceptionV3": InceptionV3,
        "MobileNet": MobileNet,
        "MobileNetV2": MobileNetV2,
        "NASNetLarge": NASNetLarge,
        "NASNetMobile": NASNetMobile,
        "ResNet50": ResNet50,
        "ResNet50V2": ResNet50V2,
        "ResNet101": ResNet101,
        "ResNet101V2": ResNet101V2,
        "ResNet152": ResNet152,
        "ResNet152V2": ResNet152V2,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "Xception": Xception
    }

    @staticmethod
    def list_all_models():
        """Returns a list of all available Keras models."""
        return KerasCatalogService.MODELS_DICT

    @staticmethod
    def list_all_strategies():
        """Returns a list of all available Keras models."""
        return [
            "Single GPU",
            "GPU Cluster Parameter Server",
            "GPU Cluster Custom",
        ]

    @staticmethod
    def list_all_layers():
        """Returns a list of all available Keras layers."""
        layers_module = tf.keras.layers
        return [cls for cls in dir(layers_module) if isinstance(getattr(layers_module, cls), type)]

    @staticmethod
    def list_all_optimizers():
        """Returns a list of all available Keras optimizers."""
        optimizers_module = tf.keras.optimizers
        return [cls for cls in dir(optimizers_module) if isinstance(getattr(optimizers_module, cls), type)]

    @staticmethod
    def get_model_object(algo_name):
        input_shape = (150, 150, 3)  # You can adjust this based on your requirement

        if algo_name not in KerasCatalogService.MODELS_DICT:
            raise ValueError(
                f"Algorithm '{algo_name}' is not available. Choose from {list(KerasCatalogService.MODELS_DICT)}")

        model_constructor = KerasCatalogService.MODELS_DICT[algo_name]
        base_model = model_constructor(weights='imagenet', include_top=False, input_shape=input_shape)
        return base_model


def init_all_models():
    for algo_name in KerasCatalogService.MODELS_DICT:
        print(algo_name)
        if algo_name == "NASNetLarge":
            input_shape = (331, 331, 3)
        else:
            input_shape = (224, 224, 3)
        model_constructor = KerasCatalogService.MODELS_DICT[algo_name]
        base_model = model_constructor(weights='imagenet', include_top=False, input_shape=input_shape)
        print("=" * 100)


if __name__ == '__main__':
    init_all_models()
