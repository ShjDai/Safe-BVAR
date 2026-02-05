import os

# from attack import add_rayleigh_noise
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import random
import time
import torch
# torch.cuda.set_device(2)
import cv2
import numpy as np
from tools.run_infinity import *
from transformers import get_cosine_schedule_with_warmup
from secret import alice, bob, generate_random_bits, compare_bit_acc, bytes_to_binary_list
from attack import add_rayleigh_noise, add_salt_and_pepper_noise, jpeg_attack

import skimage

dataset_name = 'ms-coco'

model_path='./weights/infinity_2b_reg.pth'
vae_path='./weights/infinity_vae_d32_reg.pth'
text_encoder_ckpt = 'google/flan-t5-xl'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='./cache',
    checkpoint_type='torch',
    seed=None,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=True
)


# load vae
print('load vae...')
vae = load_visual_tokenizer(args)
# load text encoder
print('load t5...')
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load infinity
print('load infinity...')
infinity = load_transformer(vae, args)

dis = torch.distributions.Bernoulli(0.5)
size = (1, 32, 1, 64, 64)
# stega_codes = (dis.sample(size).cuda() - 0.5)*2*0.1768

# os.makedirs(f'./test/{prompt_}', exist_ok=True)
cfg = 3
tau = 0.5
h_div_w = 1/1 # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

common_kwargs = {
    "g_seed": seed,
    "gt_leak": 0,
    "gt_ls_Bl": None,
    "cfg_list": cfg,
    "tau_list": tau,
    "scale_schedule": scale_schedule,
    "cfg_insertion_layer": [args.cfg_insertion_layer],
    "vae_type": args.vae_type,
    "sampling_per_bits": args.sampling_per_bits,
    "enable_positive_prompt": enable_positive_prompt
}

def gen_img_stages(
    infinity_test: Infinity, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    stega_codes=None,
    stega_mask=None
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    # print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda:0', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        stages, img_list, summed_codes_list = infinity_test.autoregressive_infer_edit_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits
        )
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img, stages, summed_codes_list


def write_bgr_img(img: torch.Tensor, file_path: str, verbose=False):
    os.makedirs(osp.dirname(osp.abspath(file_path)), exist_ok=True)
    cv2.imwrite(file_path, img.cpu().numpy())
    if verbose:
        print(f'Save to {osp.abspath(file_path)}')


    
def next_bit(bits: list):
    for b in bits:
        yield b


def rearrage_bits(bits: list[int], from_idx:int):
    return bits[from_idx:] + bits[:from_idx]


# prompts =[
#     "black t-rex",
#     "sun over the rainbow",
#     "chinese dragon",
#     "rats",
#     "jesus",
#     "cat",
#     "dog",
#     "bird",
#     "fish",
#     "horse",
#     "cow",
#     "sun shines on the sea",
#     "rain drops on the window",
#     "a puma at large",
#     "indian man",
#     "astronaut over the moon",
#     "happy hippo in the jungle",
#     "a cat in the hat",
#     "a snake in the grass",
#     "lemon tree",
#     "computer science",
#     "islamic art",
#     "a man is walking on the street",
#     "tender moonlight",
#     "whale in the ocean"
# ]

with open('/root/autodl-tmp/repo/mas_GRDH/text_prompt_dataset/coco_dataset.txt', 'r') as f:
    prompts = f.readlines()

if len(prompts[-1]) == 0:
    prompts = prompts[:-1]

# secret = b'qwert world. very long ago, there was a cat. salam. nihao. asdfg' * 256
# print('capacity:', len(bytes_to_binary_list(secret)))
# key = generate_random_bits(256)
# stega_bits, nonce = alice(secret, key)

# stega_codes = torch.tensor(stega_bits).cuda(0).reshape(*size)
# stega_codes = (stega_codes - 0.5) * 2 * 0.1768
# print(stega_codes.shape, stega_codes.flatten()[:10])


stega_mask = torch.Tensor([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,1,0],
    [0,0,0,0]
    ]).repeat((32,16,16)).unsqueeze(0).unsqueeze(-3).cuda()

hs = []
f = open(f'/root/autodl-tmp/data/{dataset_name}/water256/random-bits.txt', mode="a+")
for idx, prompt in enumerate(prompts):
    # prompt = "The sun sets behind the mountains, painting the sky orange"
    # prompt_ = prompt
    # prompt_ = prompt_.replace(' ', '_').replace('\n', '')
    common_kwargs["g_seed"] = idx
    # for p in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1, 0.5]:

    # ------------bob------------
    img, stages, summed_codes_list = gen_img_stages(infinity, vae, text_tokenizer, text_encoder, prompt, **common_kwargs)
    base_save_file = f'/root/autodl-tmp/data/{dataset_name}/raw/{idx}.png'
    write_bgr_img(img, base_save_file)

    # debug_codes = summed_codes_list[-1]
    # print(debug_codes.shape)
    

    # mask_dis = torch.distributions.Bernoulli(p)
    # stega_mask = mask_dis.sample(size).cuda(0)

    origin_bits = generate_random_bits(256)
    # with open('./test.txt', mode="a+") as f:
    f.write(origin_bits.hex())
    f.write('\n')
    
    
    bits = bytes_to_binary_list(origin_bits)
    # print(bits)

    stega_codes = torch.zeros((32,64,64))

    
    for c in range(32):
        rearraged_bits = rearrage_bits(bits, c*8)
        bit_generator = next_bit(rearraged_bits)    
        for i in range(64):
            for j in range(64):
                if stega_mask[0][0][0][i][j]:
                    nb = next(bit_generator)
                    stega_codes[c][i][j] = (nb - 0.5) * 2 * 0.1768

    stega_codes = stega_codes.unsqueeze(0).unsqueeze(-3).cuda()
    # stega_codes = stega_codes.repeat(32,1,1).unsqueeze(0).unsqueeze(-3).cuda()

    # ------------alice------------
    generated_image, summed_codes = gen_one_img_ste(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=idx,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
        # stega=True,
        stega_codes=stega_codes,
        stega_mask = stega_mask
    )
    save_file_stega = f'/root/autodl-tmp/data/{dataset_name}/water256/pure/{idx}.png'
    # save_file_stega = f'./{prompt_}.png'
    write_bgr_img(generated_image, save_file_stega)
    # ------------alice------------
    
    
    
    # ------------eve------------
    # add_rayleigh_noise(save_file_stega, 0.1, save_file_stega)
    # add_salt_and_pepper_noise(save_file_stega, 0.005, 0.005, save_file_stega)
    for q in [30, 40, 50, 60, 70, 80, 90, 95]:
        jpeg_attack(save_file_stega, f'/root/autodl-tmp/data/{dataset_name}/water256/jpeg{q}/{idx}.jpg', q)
    continue

    assert 0, 'stop here'

    # ------------eve------------



    # ------------bob------------
    generate_img_raw = cv2.imread(save_file_stega)[:,:,::-1]
    # generate_img_raw[600:1000, 300:900, :] = 128
    # generate_img_raw[:, :, :] = 0
    # generate_img_raw[:, :200, :] = 0
    # generate_img_raw[:, -200:, :] = 0
    # generate_img_raw = (generate_img_raw / 255 - 0.5) * 2
    # generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.0025)

    generate_img = torch.Tensor(generate_img_raw.copy())
    # generate_img = (generate_img.cuda(0) / 255 - 0.5) * 2
    # # # print(generate_img.shape)
    output = generate_img.cuda().permute(2,0,1).unsqueeze(0)
    rec = vae.forward_debug(output)

    # # # rec_img = (torch.clamp(rec[0][0], min=-1, max=1) /2 + 0.5) * 255
    # # # rec_img = rec_img.permute(1,2,0).cpu().numpy()
    z2 = rec[1]["h"].cuda(0)

    # # print(z2.shape)
    # # i = 16
    # # j = 16
    
    # z2 = stega_codes.squeeze(-3)
    # print(z2.shape)
    recover_bits = []
    for i in range(16):
        for j in range(16):
            # i = i_ * 4 + 2
            # j = j_ * 4 + 2
            bit_idx = i * 16 + j
            
            NUM = 32
            # es = [z2[0][c][bit_idx] for c in range(NUM)]
            
            es = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
            avg = sum(es)/len(es)
            # es = [e[i][j] for e in z2[0][:NUM]]
            # avg = sum(es)/len(es)
            # print('0,0:', avg)
            es2 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
            avg2 = sum(es2)/len(es2)
            # print('0,1:', avg2)
            es3 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
            avg3 = sum(es3)/len(es3)
            # print('1,0:', avg3)
            es4 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
            avg4 = sum(es4)/len(es4)
            # print('1,1:', avg4)
            es5 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
            avg5 = sum(es5)/len(es5)
            # print('-1,0:', avg5)
            es6 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
            avg6 = sum(es6)/len(es6)
            # # print('-1,1:', avg6)
            es7 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
            avg7 = sum(es7)/len(es7)
            # # print('0,-1:', avg7)
            es8 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
            avg8 = sum(es8)/len(es8)
            # # print('1,-1:', avg8)
            es9 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
            avg9 = sum(es9)/len(es9)
            avg_avg = sum([avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9]) / 8
            # avg_avg = sum([avg2, avg3, avg4]) / 3
            if avg_avg > avg:
                recover_bits.append(0)
            else:
                recover_bits.append(1)
    # # print(recover_bits[:30])
    # # print(bits[:30])
    err_num = 0
    for b1, b2 in zip(recover_bits, bits):
        if b1 != b2:
            err_num += 1
    print(err_num / 256)
    hs.append(err_num / 256)

    # print('-1,-1:', avg9)
    # print(avg,sum([avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9]) / 8)
    # decode_stega = z2.unsqueeze(-3) - summed_codes_list[-2]
    # positives = torch.ones_like(decode_stega)
    # negatives = torch.zeros_like(decode_stega)
    # decode_stega_01 = torch.where(decode_stega > 0, positives, negatives).flatten().tolist()
    # # print(len(decode_stega_01))
    # plain_text = bob([int(b) for b in decode_stega_01], nonce, key)
    # # print(plain_text)
    # acc = compare_bit_acc(secret, plain_text)
    # print(f'acc: {acc}')
    # hs.append(acc)

f.close()
# print(np.average(hs))
