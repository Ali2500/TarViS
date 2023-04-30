import tarvis.modelling.backbone.build_resnet as resnet
import tarvis.modelling.backbone.build_swin as swin
import tarvis.modelling.backbone.build_convnext as convnext


backbone_builder = {
    "ResNet50": resnet.build_resnet50_backbone,
    "ResNet101": resnet.build_resnet101_backbone,
    "SwinTiny": swin.build_swin_tiny_backbone,
    "SwinSmall": swin.build_swin_small_backbone,
    "SwinBase": swin.build_swin_base_backbone,
    "SwinLarge": swin.build_swin_large_backbone,
    "ConvNextTiny": convnext.convnext_tiny,
    "ConvNextSmall": convnext.convnext_small,
    "ConvNextBase": convnext.convnext_base,
    "ConvNextLarge": convnext.convnext_large,
    "ConvNextXLarge": convnext.convnext_xlarge
}


