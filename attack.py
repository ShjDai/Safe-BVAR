import cv2
import torch
import numpy as np
import random
import skimage
from PIL import Image, ImageFilter


def add_salt_and_pepper_noise(file_path: str, salt_prob: float, pepper_prob: float, save_path: str):
    image = cv2.imread(file_path)
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    
    # 添加盐噪声
    num_salt = int(salt_prob * h * w)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255  # 对所有通道添加盐噪声
 
    # 添加胡椒噪声
    num_pepper = int(pepper_prob * h * w)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0  # 对所有通道添加胡椒噪声
 
    cv2.imwrite(save_path, noisy_image)


def add_rayleigh_noise(file_path: str, sigma: float, save_path: str):
    """
    对图像添加瑞利噪声。
    
    参数:
        image (ndarray): 输入图像，范围 [0, 255]。
        sigma (float): 瑞利分布的尺度参数（控制噪声强度）。
        
    返回:
        noisy_image (ndarray): 添加噪声后的图像。
    """
    image = cv2.imread(file_path)
    # 归一化图像到 [0, 1]
    normalized_image = image / 255.0
 
    # 生成瑞利噪声
    rayleigh_noise = np.random.rayleigh(scale=sigma, size=normalized_image.shape)
 
    # 添加噪声并裁剪到 [0, 1]
    noisy_image = np.clip(normalized_image + rayleigh_noise, 0, 1)
 
    # 恢复到 [0, 255] 范围并转换为 uint8 类型
    cv2.imwrite(save_path, (noisy_image * 255).astype(np.uint8))


def jpeg_attack(file_path: str, save_path: str, quality: int = 90):
    img_gt = cv2.imread(file_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_gt, encode_param)   # 以JPEG格式进行编码
    img_lq = np.float32(cv2.imdecode(encimg, 1))            # 解码编码后的图像，并将其转换为浮点类型
    cv2.imwrite(save_path, img_lq) 


def attacks(generate_img_raw, attack_type: str):
    match attack_type:
        case 'rd50':
            drop_h = random.randint(512, 1024)
            drop_w = (1024*1024//2) // drop_h
            drop_i = random.randint(0, 1024-drop_h)
            drop_j = random.randint(0, 1024-drop_w)
            generate_img_raw[drop_i:drop_i+drop_h, drop_j:drop_j+drop_w, :] = 0    
            # print(drop_i,drop_i+drop_h, drop_j,drop_j+drop_w)
        case 'cd256':
            generate_img_raw[384:640,384:640,:] = 0
        case 'cd512':
            generate_img_raw[256:768,256:768,:] = 0
        case 'cd856':
            generate_img_raw[84:940,84:940,:] = 0
        case 'cd724':
            generate_img_raw[150:874,150:874,:] = 0
        case 'cd792':
            generate_img_raw[116:908,116:908,:] = 0
        case 'cd916':
            generate_img_raw[54:970,54:970,:] = 0
        case 'rs4':
            rss = 256
            generate_img_raw = cv2.resize(generate_img_raw, (rss,rss))
            generate_img_raw = cv2.resize(generate_img_raw, (1024,1024))
        case 'g001':
            generate_img_raw = (np.array(generate_img_raw).astype(np.uint8) / 255 - 0.5) * 2
            generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.01)
            # generate_img_raw = np.array(generate_img_raw * 255, dtype=np.uint8)
            generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
        case 'g003':
            # generate_img_raw = (np.array(generate_img_raw).astype(np.uint8) / 255 - 0.5) * 2
            generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.075**2)
            generate_img_raw = np.array(generate_img_raw * 255, dtype=np.uint8)
            # generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
        case 'gb2':
            generate_img_raw = (generate_img_raw / 255 - 0.5) * 2
            generate_img_raw = skimage.filters.gaussian(generate_img_raw, sigma=2)
            generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
        case 'gbr2':
            img = Image.fromarray(generate_img_raw)
            generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=2))
            generate_img_raw = np.array(generate_img_raw)
        case 'gbr3':
            img = Image.fromarray(generate_img_raw)
            generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=3))
            generate_img_raw = np.array(generate_img_raw)
        case 'gbr4':
            img = Image.fromarray(generate_img_raw)
            generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=4))
            generate_img_raw = np.array(generate_img_raw)
        case 'gbr5':
            img = Image.fromarray(generate_img_raw)
            generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=5))
            generate_img_raw = np.array(generate_img_raw)
        case 'gbr1':
            img = Image.fromarray(generate_img_raw)
            generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=1))
            generate_img_raw = np.array(generate_img_raw)
        case 'jpeg90':
            # img_gt = cv2.imread(file_path)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encimg = cv2.imencode('.jpg', generate_img_raw, encode_param)   # 以JPEG格式进行编码
            generate_img_raw = np.float32(cv2.imdecode(encimg, 1)).astype(np.uint8)       # 解码编码后的图像，并将其转换为浮点类型
    return generate_img_raw


if __name__ == "__main__":
    # add_salt_and_pepper_noise("/groupshare_1/dsj/Infinity/ipynb_tmp.jpg", 0.02, 0.02, "/groupshare_1/dsj/Infinity/ipynb_tmp_salt_and_pepper.jpg")
    jpeg_attack("/groupshare_1/dsj/Infinity/ipynb_tmp.jpg", "/groupshare_1/dsj/Infinity/ipynb_tmp_jpeg.jpg")