from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing


# This code has been inspired by https://github.com/ehuynh1106/TinyImageNet-Transformers, the writers of the paper
# Vision Transformers in 2022: An update on tiny ImageNet. arXiv (Cornell University).
# https://doi.org/10.48550/arxiv.2205.10660 (Huynh, E. (2022b).)
def get_data_augmentations(num_classes, label_smoothing=0.1, mixup_alpha=0.8, cutmix_alpha=1.0, rand_erase=0.25):
    if mixup_alpha or cutmix_alpha:
        mixup = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            label_smoothing=label_smoothing,
            num_classes=num_classes
        )
    if rand_erase:
        random_erase = RandomErasing(
            probability=rand_erase,
            mode='pixel'
        )

    return mixup, random_erase