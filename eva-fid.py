import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


# YOU SHOULD SUBSTITUTE ... WITH YOUR LOCAL FOLDER
real_images_folder = ...  # folder with pure img
generated_images_folder = ...  # folder with watermarked img, without attack

inception_model = torchvision.models.inception_v3(pretrained=True)

# FID
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                batch_size=20, device='cuda', dims=2048, num_workers=16)
print('FID value:', fid_value)

