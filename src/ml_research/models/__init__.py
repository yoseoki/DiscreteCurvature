from ml_research.models import cnn as nmc
from itertools import chain as _chain
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model registry (custom)
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
	"UNET": {
		"build": lambda nc: nmc.UNET(nc),
		"non_save_layers": ["bias", ".2.w", ".2.b", ".5.w", ".5.b"],
	},
	"RESNET50": {
		"build": lambda nc: nmc.RESNET50(in_channels=3, num_classes=nc),
		"non_save_layers": ["bias", "iden", "batchnorm"],
	},
	"RESNET18": {
		"build": lambda nc: nmc.RESNET18(nc),
		"non_save_layers": ["bias", "downsample.1", "bn"],
	},
	"RESNET18_SMALL": {
		"build": lambda nc: nmc.RESNET18_small(nc),
		"non_save_layers": ["bias", "downsample.1", "bn"],
	},
	"VGG16": {
		"build": lambda nc: nmc.VGG16(64, num_classes=nc),
		"non_save_layers": ["bias", ".1.w", ".1.b", ".4.w", ".4.b", ".7.w", ".7.b"],
	},
	"VIT": {
		"build": lambda nc: nmc.ViT(
			in_channels=3, patch_size=4, emb_size=192,
			img_size=32, depth=12, n_classes=nc,
		),
		"non_save_layers": ["bias", "fn.0.weight"],
	},
	"LENET": {
		"build": lambda nc: nmc.LeNet(num_classes=nc),
		"non_save_layers": ["bias"],
	},
	"LENET_REVISED": {
		"build": lambda nc: nmc.LeNetRevised(num_classes=nc),
		"non_save_layers": ["bias"],
	},
}

# ---------------------------------------------------------------------------
# Model registry (timm)
# ---------------------------------------------------------------------------
_TIMM_REGISTRY = {
	"VGG16": {
		"timm_name": "vgg16",
		"timm_kwargs": {},
		"non_save_layers": ["bias"],
	},
	"RESNET18": {
		"timm_name": "resnet18",
		"timm_kwargs": {},
		"non_save_layers": ["bias", "downsample.1", "bn", ".conv1."],
	},
	"RESNET50": {
		"timm_name": "resnet50",
		"timm_kwargs": {},
		"non_save_layers": ["bias", "bn"],
	},
	"VIT": {
		"timm_name": "vit_tiny_patch16_224",
		"timm_kwargs": {"img_size": 32, "patch_size": 4},
		"non_save_layers": ["bias", "norm"],
	},
}

# ---------------------------------------------------------------------------
# build_model / get_non_save_layers
# ---------------------------------------------------------------------------

def build_model(model_name, num_classes, backend="timm"):
	name = model_name.upper()

	if backend == "timm" and name in _TIMM_REGISTRY:
		import timm
		entry = _TIMM_REGISTRY[name]
		return timm.create_model(
			entry["timm_name"],
			pretrained=True,
			num_classes=num_classes,
			**entry["timm_kwargs"],
		)

	if name not in _MODEL_REGISTRY:
		raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
	return _MODEL_REGISTRY[name]["build"](num_classes)


def get_non_save_layers(model_name, backend="timm"):
	name = model_name.upper()

	if backend == "timm" and name in _TIMM_REGISTRY:
		return _TIMM_REGISTRY[name]["non_save_layers"]

	if name not in _MODEL_REGISTRY:
		raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
	return _MODEL_REGISTRY[name]["non_save_layers"]

# ---------------------------------------------------------------------------
# Layer groups for layerwise optimizers
# ---------------------------------------------------------------------------

def _custom_vgg16_groups(model):
	names = ["feature.0.3", "feature.1.3", "feature.2.3", "feature.3.3", "feature.4.3", "connect_fc"]
	groups = [
		model.feature[0].parameters(),
		model.feature[1].parameters(),
		model.feature[2].parameters(),
		model.feature[3].parameters(),
		model.feature[4].parameters(),
		model.connect_fc.parameters(),
	]
	return names, groups


def _custom_resnet18_small_groups(model):
	names = ["conv1", "layer11.conv2", "layer21.conv2", "layer31.conv2", "layer41.conv2", "fc"]
	groups = [
		_chain(model.conv1.parameters(), model.bn1.parameters()),
		_chain(model.layer11.parameters(), model.layer12.parameters()),
		_chain(model.layer21.parameters(), model.layer22.parameters()),
		_chain(model.layer31.parameters(), model.layer32.parameters()),
		_chain(model.layer41.parameters(), model.layer42.parameters()),
		model.fc.parameters(),
	]
	return names, groups


def _custom_resnet18_groups(model):
	names = ["conv1", "layer11.conv2", "layer21.conv2", "layer31.conv2", "layer41.conv2", "fc"]
	groups = [
		_chain(model.conv1.parameters(), model.bn1.parameters()),
		_chain(model.layer11.parameters(), model.layer12.parameters()),
		_chain(model.layer21.parameters(), model.layer22.parameters()),
		_chain(model.layer31.parameters(), model.layer32.parameters()),
		_chain(model.layer41.parameters(), model.layer42.parameters()),
		model.fc.parameters(),
	]
	return names, groups


def _custom_resnet50_groups(model):
	names = ["conv1", "conv2_x.1.convseq.1.conv", "conv3_x.1.convseq.1.conv",
			 "conv4_x.1.convseq.1.conv", "conv5_x.0.convseq.1.conv", "fc"]
	groups = [
		model.conv1.parameters(),
		model.conv2_x.parameters(),
		model.conv3_x.parameters(),
		model.conv4_x.parameters(),
		model.conv5_x.parameters(),
		model.fc.parameters(),
	]
	return names, groups


def _custom_vit_groups(model):
	names = ["0.positions"]
	groups = [model[0].parameters()]
	for i in range(12):
		names.append(f"1.{i}.1.fn.1.0.weight")
		groups.append(model[1][i].parameters())
	names.append("2.2.weight")
	groups.append(model[2].parameters())
	return names, groups


def _custom_lenet_groups(model):
	names = ["cnn1.weight", "cnn2.weight", "fc1.weight", "fc2.weight", "fc3.weight"]
	groups = [
		model.cnn1.parameters(),
		model.cnn2.parameters(),
		model.fc1.parameters(),
		model.fc2.parameters(),
		model.fc3.parameters(),
	]
	return names, groups


def _timm_resnet_groups(model):
	names = ["conv1", "layer1.0.conv2", "layer2.0.conv2", "layer3.0.conv2", "layer4.0.conv2", "fc"]
	groups = [
		_chain(model.conv1.parameters(), model.bn1.parameters()),
		model.layer1.parameters(),
		model.layer2.parameters(),
		model.layer3.parameters(),
		model.layer4.parameters(),
		model.fc.parameters(),
	]
	return names, groups


def _timm_vgg16_groups(model):
	features = list(model.features.children())
	blocks, current = [], []
	for layer in features:
		current.append(layer)
		if isinstance(layer, nn.MaxPool2d):
			blocks.append(current)
			current = []
	if current:
		blocks.append(current)

	names = [f"features_block{i}" for i in range(len(blocks))]
	groups = []
	for block in blocks:
		params = []
		for layer in block:
			params.extend(layer.parameters())
		groups.append(iter(params))

	names.append("head")
	groups.append(_chain(model.pre_logits.parameters(), model.head.parameters()))
	return names, groups


def _timm_vit_groups(model):
	names = ["patch_embed"]
	groups = [model.patch_embed.parameters()]
	for i in range(len(model.blocks)):
		names.append(f"blocks.{i}")
		groups.append(model.blocks[i].parameters())
	names.append("head")
	groups.append(model.head.parameters())
	return names, groups


_CUSTOM_LAYER_GROUPS = {
	"VGG16": _custom_vgg16_groups,
	"RESNET18_SMALL": _custom_resnet18_small_groups,
	"RESNET18": _custom_resnet18_groups,
	"RESNET50": _custom_resnet50_groups,
	"VIT": _custom_vit_groups,
	"LENET": _custom_lenet_groups,
}

_TIMM_LAYER_GROUPS = {
	"VGG16": _timm_vgg16_groups,
	"RESNET18": _timm_resnet_groups,
	"RESNET50": _timm_resnet_groups,
	"VIT": _timm_vit_groups,
}


def get_layer_groups(model_name, model, backend="timm"):
	name = model_name.upper()
	if backend == "timm" and name in _TIMM_LAYER_GROUPS:
		return _TIMM_LAYER_GROUPS[name](model)
	if name not in _CUSTOM_LAYER_GROUPS:
		raise ValueError(f"No layer groups defined for model: {name}")
	return _CUSTOM_LAYER_GROUPS[name](model)
