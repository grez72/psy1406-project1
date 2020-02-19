import pandas as pd
from scipy.stats import pearsonr
from fastprogress import master_bar, progress_bar
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from . import nethook as nethook
from .data import ImageListDataset
import torch


def get_layer(m, layers):
    layer = layers.pop(0)
    m = getattr(m, layer)
    if len(layers) > 0:
        return get_layer(m, layers)
    return m


def get_layers(model, parent_name='', layer_info=[]):
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            layer_info = get_layers(module, layer_name, layer_info=layer_info)
        else:
            layer_info.append(layer_name.strip('.'))

    return layer_info


def get_layer_type(model, layer_name):
    m = get_layer(model, layer_name.split("."))
    return m.__class__.__name__


def convert_relu_layers(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu_layers(child)


def store_activations(model, layer_names):
    a = OrderedDict()
    for layer_num, layer_name in enumerate(layer_names):
        layer_type = get_layer_type(model.model, layer_name)
        X = model.retained_layer(layer_name)
        X = X.view(X.shape[0], -1)
        a[layer_name] = X
    return a


def compute_similarity(model, dataset):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hook model
    layer_names = get_layers(model, parent_name='', layer_info=[])
    if not isinstance(model, nethook.InstrumentedModel):
        model = nethook.InstrumentedModel(model)
    for layer_name in layer_names:
        model.retain_layer(layer_name)

    model = model.to(device)
    model.eval()

    # create dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageListDataset(imgs=dataset.files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0, pin_memory=False)

    # compute similarity by layer
    df = pd.DataFrame(columns=['pair_num', 'image1', 'image2',
                               'layer_num', 'layer_name', 'layer_type', 'r'])
    pair_num = 0
    mb = master_bar(dataloader)
    for count, (imgs, labels, indexes) in enumerate(mb):
        with torch.no_grad():
            model(imgs.to(device))
        if count % 2 == 0:
            a1 = store_activations(model, layer_names)
            image1 = dataset.files[indexes].name
        if count % 2 == 1:
            a2 = store_activations(model, layer_names)
            image2 = dataset.files[indexes].name
            for layer_num, layer_name in enumerate(progress_bar(layer_names, parent=mb)):
                r = pearsonr(a1[layer_name].squeeze(),
                             a2[layer_name].squeeze())[0]
                layer_type = get_layer_type(model.model, layer_name)
                df = df.append({
                    "pair_num": pair_num,
                    "image1": image1,
                    "image2": image2,
                    "layer_num": layer_num,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "r": r,
                }, ignore_index=True)

            pair_num += 1

    df.pair_num = df.pair_num.astype(int)

    return df
