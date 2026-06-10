from ml_research.training import trainer
import sys
from random import shuffle

myTrainer = trainer.CIFAR10Trainer()

print("="*40)
print("1. 의미가 있는 방향 ")
print("="*40)

for indicator in range(1, 10):
    seed = int(sys.argv[2])
    epoch = indicator * 5
    model_name = sys.argv[1]
    config_file_path = "/home/yoseok/ModelTrainer/config/config_{}_{}.json".format(model_name, sys.argv[3])
    weightPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/weights_epoch{:03d}_all.pt".format(seed, epoch)
    gradPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/gradients_epoch{:03d}_all.safetensors".format(seed, epoch)
    saltedMagPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/salted_mag.csv".format(seed)
    with open(saltedMagPath, 'r') as f:
        lines = f.readlines()
    rawSaltedMag = lines[indicator-1].split(",")
    saltedMag = []
    for rawele in rawSaltedMag:
        saltedMag.append(float(rawele)) # target
    is_verbose = False
    myTrainer.validate_geodesic(model_name, seed, config_file_path, weightPath, gradPath, saltedMag, is_verbose=is_verbose, prefix="[indicator : {:02d}]".format(indicator))

print("="*40)
print("2. 의미가 없는 방향(shuffle) ")
print("="*40)
idx = [0, 1, 2, 3, 4, 5]
shuffle(idx)
print("idx : ", idx)

for indicator in range(1, 10):
    seed = int(sys.argv[2])
    epoch = indicator * 5
    model_name = sys.argv[1]
    config_file_path = "/home/yoseok/ModelTrainer/config/config_{}_{}.json".format(model_name, sys.argv[3])
    weightPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/weights_epoch{:03d}_all.pt".format(seed, epoch)
    gradPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/gradients_epoch{:03d}_all.safetensors".format(seed, epoch)
    saltedMagPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/salted_mag.csv".format(seed)
    with open(saltedMagPath, 'r') as f:
        lines = f.readlines()
    rawSaltedMag = lines[indicator-1].split(",")
    saltedMag = []
    for rawele in rawSaltedMag:
        saltedMag.append(float(rawele)) # target
    is_verbose = False
    newSaltedMag = []
    for id in idx:
        newSaltedMag.append(saltedMag[id])
    myTrainer.validate_geodesic(model_name, seed, config_file_path, weightPath, gradPath, newSaltedMag, is_verbose=is_verbose, prefix="[indicator : {:02d}]".format(indicator))

print("="*40)
print("3. 방향 변환 없이 ")
print("="*40)

for indicator in range(1, 10):
    seed = int(sys.argv[2])
    epoch = indicator * 5
    model_name = sys.argv[1]
    config_file_path = "/home/yoseok/ModelTrainer/config/config_{}_{}.json".format(model_name, sys.argv[3])
    weightPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/weights_epoch{:03d}_all.pt".format(seed, epoch)
    gradPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/gradients_epoch{:03d}_all.safetensors".format(seed, epoch)
    saltedMagPath = "/home/yoseok/ModelTrainer/resnet18_small_1_flatten/RESNET18_SMALL__{:04d}/salted_mag.csv".format(seed)
    with open(saltedMagPath, 'r') as f:
        lines = f.readlines()
    rawSaltedMag = lines[indicator-1].split(",")
    saltedMag = []
    for rawele in rawSaltedMag:
        # saltedMag.append(float(rawele)) # target
        saltedMag.append(2.0) # control
    is_verbose = False
    myTrainer.validate_geodesic(model_name, seed, config_file_path, weightPath, gradPath, saltedMag, is_verbose=is_verbose, prefix="[indicator : {:02d}]".format(indicator))
