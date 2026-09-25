from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class LEVIRCDPatchDataset(Dataset):
    def __init__(self, root_dir, mask_transform, max_samples=10192, image_size = 256):

        self.dir_A = os.path.join(root_dir, "A")
        self.dir_B = os.path.join(root_dir, "B")
        self.dir_label = os.path.join(root_dir, "label")

        all_files = os.listdir(self.dir_A)
        png_files = [f for f in all_files if f.lower().endswith('.png')]
        self.filenames = sorted(png_files)[:max_samples]

        # Same normalization used by SegFormer
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = mask_transform
        self.batch_size = 8

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        im1 = Image.open(
            os.path.join(self.dir_A, filename)
        ).convert("RGB")

        im2 = Image.open(
            os.path.join(self.dir_B, filename)
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.dir_label, filename)
        ).convert("L")

        im1 = self.image_transform(im1)
        im2 = self.image_transform(im2)

        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return im1, im2, mask