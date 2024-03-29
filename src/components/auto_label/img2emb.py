# Code reference: https://github.com/christiansafka/img2vec.git
# License: MIT

import os

import torch
import torchvision.models as models
import torchvision.transforms as transforms


class Img2Emb:
    RESNET_OUTPUT_SIZES = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    EFFICIENTNET_OUTPUT_SIZES = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
    }

    def __init__(self, model="resnet-18", layer="default", layer_output_size=512, save=False):
        """Img2Emb
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        if save:
            torch.save(self.model, os.path.join("..", "model", f"{self.model_name}.pt"))
            # export onnx
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                os.path.join("..", "model", f"{self.model_name}.onnx"),
                verbose=True,
                input_names=["input"],
                output_names=["output"],
            )

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_emb(self, img, tensor=False):
        """Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if isinstance(img, list):
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ["alexnet", "vgg"]:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ["alexnet", "vgg"]:
                    return my_embedding.numpy()[:, :]
                elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            if self.model_name in ["alexnet", "vgg"]:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name == "densenet" or "efficientnet" in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ["alexnet", "vgg"]:
                    return my_embedding.numpy()[0, :]
                elif self.model_name == "densenet":
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def get_embs(self, imgs, tensor=False):
        """Get vector embeddings from list of PIL images
        :param imgs: List of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        # TODO: batch this
        return [self.get_emb(img, tensor) for img in imgs]

    def _get_model_and_layer(self, model_name, layer):
        """Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name.startswith("resnet") and not model_name.startswith("resnet-"):
            model = getattr(models, model_name)(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == "resnet-18":
            model = models.resnet18(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            if layer == "default":
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == "vgg":
            # VGG-11
            model = models.vgg11_bn(pretrained=True)
            if layer == "default":
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features  # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == "densenet":
            # Densenet-121
            model = models.densenet121(pretrained=True)
            if layer == "default":
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features  # should be 1024
            else:
                raise KeyError(f"Un support {model_name} for layer parameters")

            return model, layer

        elif "efficientnet" in model_name:
            # efficientnet-b0 ~ efficientnet-b7
            if model_name == "efficientnet_b0":
                model = models.efficientnet_b0(pretrained=True)
            elif model_name == "efficientnet_b1":
                model = models.efficientnet_b1(pretrained=True)
            elif model_name == "efficientnet_b2":
                model = models.efficientnet_b2(pretrained=True)
            elif model_name == "efficientnet_b3":
                model = models.efficientnet_b3(pretrained=True)
            elif model_name == "efficientnet_b4":
                model = models.efficientnet_b4(pretrained=True)
            elif model_name == "efficientnet_b5":
                model = models.efficientnet_b5(pretrained=True)
            elif model_name == "efficientnet_b6":
                model = models.efficientnet_b6(pretrained=True)
            elif model_name == "efficientnet_b7":
                model = models.efficientnet_b7(pretrained=True)
            else:
                raise KeyError(f"Un support {model_name}.")

            if layer == "default":
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError(f"Un support {model_name} for layer parameters")

            return model, layer

        else:
            raise KeyError(f"Model {model_name} was not found")
