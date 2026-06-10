import hydra
from omegaconf import DictConfig
from ml_research.training import trainer

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.data.name == "cifar10":
        myTrainer = trainer.CIFAR10Trainer()
    elif cfg.data.name == "cifar100":
        myTrainer = trainer.CIFAR100Trainer()
    elif cfg.data.name == "svhn":
        myTrainer = trainer.SVHNTrainer()
    myTrainer.training(cfg, is_verbose=cfg.is_verbose)

if __name__ == "__main__":
    main()
