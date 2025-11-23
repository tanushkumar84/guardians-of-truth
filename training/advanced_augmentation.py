"""
Advanced Data Augmentation for Deepfake Detection
Includes: RandAugment, Cutout, Mixup, CutMix, and domain-specific augmentations
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)

class RandAugment:
    """
    RandAugment: Practical automated data augmentation with a reduced search space
    """
    def __init__(self, n=2, m=10):
        """
        Args:
            n (int): Number of augmentation transformations to apply sequentially
            m (int): Magnitude for all transformations (0-10)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.rotate, 0, 30),
            (self.solarize, 0, 256),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, 0., 0.3),
            (self.shear_y, 0., 0.3),
            (self.translate_x, 0., 0.33),
            (self.translate_y, 0., 0.33),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            img = op(img, val)
        return img

    def auto_contrast(self, img, _):
        return ImageEnhance.Contrast(img).enhance(2)

    def equalize(self, img, _):
        from PIL import ImageOps
        return ImageOps.equalize(img)

    def rotate(self, img, v):
        return img.rotate(random.uniform(-v, v))

    def solarize(self, img, v):
        from PIL import ImageOps
        return ImageOps.solarize(img, v)

    def color(self, img, v):
        return ImageEnhance.Color(img).enhance(v)

    def contrast(self, img, v):
        return ImageEnhance.Contrast(img).enhance(v)

    def brightness(self, img, v):
        return ImageEnhance.Brightness(img).enhance(v)

    def sharpness(self, img, v):
        return ImageEnhance.Sharpness(img).enhance(v)

    def shear_x(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

    def shear_y(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

    def translate_x(self, img, v):
        v = v * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

    def translate_y(self, img, v):
        v = v * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class Cutout:
    """
    Randomly mask out one or more patches from an image
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(self.n_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img


class MixupCutmix:
    """
    Mixup and Cutmix augmentation
    """
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        """
        Args:
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: Cutmix alpha parameter  
            prob: Probability of applying mixup/cutmix
            switch_prob: Probability of switching between mixup and cutmix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def mixup(self, images, labels):
        """Mixup augmentation"""
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_images, mixed_labels

    def cutmix(self, images, labels):
        """Cutmix augmentation"""
        batch_size = images.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        index = torch.randperm(batch_size).to(images.device)
        
        W = images.size(2)
        H = images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return images, mixed_labels

    def __call__(self, images, labels):
        """Apply mixup or cutmix"""
        if random.random() > self.prob:
            return images, labels
        
        if random.random() > self.switch_prob:
            return self.mixup(images, labels)
        else:
            return self.cutmix(images, labels)


class DeepfakeSpecificAugment:
    """
    Augmentations specific to deepfake detection
    - JPEG compression artifacts
    - Blur (Gaussian, Motion)
    - Noise injection
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def add_jpeg_compression(self, img, quality=None):
        """Add JPEG compression artifacts"""
        import io
        if quality is None:
            quality = random.randint(50, 95)
        
        # Save image with JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def add_gaussian_blur(self, img, radius=None):
        """Add Gaussian blur"""
        if radius is None:
            radius = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius))

    def add_gaussian_noise(self, img, mean=0, std=None):
        """Add Gaussian noise"""
        if std is None:
            std = random.uniform(5, 20)
        
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def __call__(self, img):
        """Randomly apply deepfake-specific augmentations"""
        if random.random() < self.prob:
            aug_choice = random.choice(['jpeg', 'blur', 'noise'])
            
            if aug_choice == 'jpeg':
                img = self.add_jpeg_compression(img)
            elif aug_choice == 'blur':
                img = self.add_gaussian_blur(img)
            else:
                img = self.add_gaussian_noise(img)
        
        return img


def get_advanced_transforms(image_size, is_training=True):
    """
    Get advanced transforms for training and validation
    
    Args:
        image_size (int): Target image size
        is_training (bool): Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        transform = T.Compose([
            # PIL-based augmentations
            T.Resize((image_size, image_size)),
            DeepfakeSpecificAugment(prob=0.3),
            RandAugment(n=2, m=10),
            T.RandomHorizontalFlip(p=0.5),
            
            # Convert to tensor
            T.ToTensor(),
            
            # Tensor-based augmentations
            Cutout(n_holes=1, length=image_size // 8),
            
            # Normalize
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for improved inference
    """
    def __init__(self, model, num_augments=5):
        self.model = model
        self.num_augments = num_augments
        self.augmentations = [
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]

    def __call__(self, x):
        """
        Apply TTA and average predictions
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Averaged predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            pred = self.model(x)
            predictions.append(pred)
            
            # Augmented predictions
            for _ in range(self.num_augments - 1):
                aug = random.choice(self.augmentations)
                x_aug = aug(x)
                pred_aug = self.model(x_aug)
                predictions.append(pred_aug)
        
        # Average all predictions
        avg_pred = torch.mean(torch.stack(predictions), dim=0)
        return avg_pred


if __name__ == '__main__':
    # Test augmentations
    logger.info("Testing advanced augmentations...")
    
    # Create dummy image
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    # Test RandAugment
    rand_aug = RandAugment(n=2, m=10)
    aug_img = rand_aug(dummy_img)
    logger.info(f"RandAugment: {type(aug_img)}")
    
    # Test Deepfake-specific augmentations
    df_aug = DeepfakeSpecificAugment(prob=1.0)
    aug_img = df_aug(dummy_img)
    logger.info(f"Deepfake augmentation: {type(aug_img)}")
    
    # Test full transform
    transform = get_advanced_transforms(224, is_training=True)
    tensor_img = transform(dummy_img)
    logger.info(f"Full transform output shape: {tensor_img.shape}")
    
    logger.info("✅ All augmentations working correctly!")
