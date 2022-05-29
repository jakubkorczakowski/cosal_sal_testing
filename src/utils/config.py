import torchvision.models as models

MODELS_TO_TEST = {
    "vgg16" : {
        "model": models.vgg16(pretrained=True),
        "return_nodes": {"features.30": "features"},
        "output_size": 512
    },
    "vgg19" : {
        "model": models.vgg19(pretrained=True),
        "return_nodes": {"features.36": "features"},
        "output_size": 512
    },
    "resnet50" : {
        "model": models.resnet50(pretrained=True),
        "return_nodes": {"layer4.2.relu_2": "features"},
        "output_size": 2048
    },
    "resnet152" : {
        "model": models.resnet152(pretrained=True),
        "return_nodes": {"layer4.2.relu_2": "features"},
        "output_size": 2048
    },
    # "efficientnet_b7" : {
    #     "model": models.efficientnet_b7(pretrained=True),
    #     "return_nodes": {"features.8.2": "features"},
    #     "output_size": 2560
    # },
    # "regnet_x_32gf" : {
    #     "model": models.regnet_x_32gf(pretrained=True),
    #     "return_nodes": {"trunk_output.block4.block4-0.activation": "features"},
    #     "output_size": 2520
    # },
    # "convnext_base" : {
    #     "model": models.convnext_base(pretrained=True),
    #     "return_nodes": {"features.7.2.add": "features"},
    #     "output_size": 1024
    # },
    # "convnext_large" : {
    #     "model": models.convnext_large(pretrained=True),
    #     "return_nodes": {"features.7.2.add": "features"},
    #     "output_size": 1536
    # }
}

# MODELS_TO_TEST = {
#     # "resnet50_41" : {
#     #     "model": models.resnet50(pretrained=True),
#     #     "return_nodes": {"layer4.1.relu_2": "features"},
#     #     "output_size": 2048
#     # },
#     # "resnet50_40" : {
#     #     "model": models.resnet50(pretrained=True),
#     #     "return_nodes": {"layer4.0.relu_2": "features"},
#     #     "output_size": 2048
#     # },
#     "resnet50_35" : {
#         "model": models.resnet50(pretrained=True),
#         "return_nodes": {"layer3.5.relu_2": "features"},
#         "output_size": 1024
#     },
#     "resnet50_32" : {
#         "model": models.resnet50(pretrained=True),
#         "return_nodes": {"layer3.2.relu_2": "features"},
#         "output_size": 1024
#     },
#     "resnet50_23" : {
#         "model": models.resnet50(pretrained=True),
#         "return_nodes": {"layer2.3.relu_2": "features"},
#         "output_size": 512
#     },
#     "resnet50_12" : {
#         "model": models.resnet50(pretrained=True),
#         "return_nodes": {"layer1.2.relu_2": "features"},
#         "output_size": 256
#     },
# }

# MODELS_TO_TEST = {
    # "vgg19_36" : {
    #     "model": models.vgg19(pretrained=True),
    #     "return_nodes": {"features.36": "features"},
    #     "output_size": 512
    # },
    # "vgg19_31" : {
    #     "model": models.vgg19(pretrained=True),
    #     "return_nodes": {"features.31": "features"},
    #     "output_size": 512
    # },
    # "vgg19_27" : {
    #     "model": models.vgg19(pretrained=True),
    #     "return_nodes": {"features.27": "features"},
    #     "output_size": 512
    # },
    # "vgg19_18" : {
    #     "model": models.vgg19(pretrained=True),
    #     "return_nodes": {"features.18": "features"},
    #     "output_size": 256
    # },
    # "vgg19_9" : {
    #     "model": models.vgg19(pretrained=True),
    #     "return_nodes": {"features.9": "features"},
    #     "output_size": 128
    # },
# }