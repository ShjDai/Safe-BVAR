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
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=True
)

# os.makedirs(f'./test/{prompt_}', exist_ok=True)
cfg = 3
tau = 0.5
h_div_w = 1.2 # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

print(h_div_w_templates)
print(h_div_w_template_)
print(scale_schedule)


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
size = (1, 32, 1, 70, 56)
# stega_codes = (dis.sample(size).cuda() - 0.5)*2*0.1768



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


prompts =[
    "black t-rex",
    "sun over the rainbow",
    "chinese dragon",
    "rats",
    "jesus",
    "cat",
    "dog",
    "bird",
    "fish",
    "horse",
    "cow",
]

secret = b'qwert world. very long ago, there was a cat. salam. nihao. asdfg' * 245
print('capacity:', len(bytes_to_binary_list(secret)))
key = generate_random_bits(256)
stega_bits, nonce = alice(secret, key)

stega_codes = torch.tensor(stega_bits).cuda(0).reshape(*size)
stega_codes = (stega_codes - 0.5) * 2 * 0.1768
# print(stega_codes.shape, stega_codes.flatten()[:10])

hs = []
for prompt in prompts:
    # prompt = "The sun sets behind the mountains, painting the sky orange"
    prompt_ = prompt
    prompt_ = prompt_.replace(' ', '_')
    # for p in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1, 0.5]:

    # ------------bob------------
    img, stages, summed_codes_list = gen_img_stages(infinity, vae, text_tokenizer, text_encoder, prompt, **common_kwargs)
    base_save_file = f'/root/autodl-tmp/data/test/{prompt_}/test.png'
    write_bgr_img(img, base_save_file)

    debug_codes = summed_codes_list[-1]
    # print(debug_codes.shape)
    # i = 16
    # j = 16
    # es = [e[0][i][j] for e in debug_codes[0]]
    # avg = sum(es)/len(es)
    # print('0,0:', avg)
    # es2 = [e[0][i][j+1] for e in debug_codes[0]]
    # avg2 = sum(es2)/len(es2)
    # print('0,1:', avg2)
    # es3 = [e[0][i+1][j] for e in debug_codes[0]]
    # avg3 = sum(es3)/len(es3)
    # print('1,0:', avg3)
    # es4 = [e[0][i+1][j+1] for e in debug_codes[0]]
    # avg4 = sum(es4)/len(es4)
    # print('1,1:', avg4)
    # es5 = [e[0][i-1][j] for e in debug_codes[0]]
    # avg5 = sum(es5)/len(es5)
    # print('-1,0:', avg5)
    # es6 = [e[0][i-1][j+1] for e in debug_codes[0]]
    # avg6 = sum(es6)/len(es6)
    # print('-1,1:', avg6)
    # es7 = [e[0][i][j-1] for e in debug_codes[0]]
    # avg7 = sum(es7)/len(es7)
    # print('0,-1:', avg7)
    # es8 = [e[0][i+1][j-1] for e in debug_codes[0]]
    # avg8 = sum(es8)/len(es8)
    # print('1,-1:', avg8)
    # es9 = [e[0][i-1][j-1] for e in debug_codes[0]]
    # avg9 = sum(es9)/len(es9)
    # print('-1,-1:', avg9)
    # print(avg,sum([avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9]) / 8)
    # ------------bob------------

    for p in [1]:
        mask_dis = torch.distributions.Bernoulli(p)
        stega_mask = mask_dis.sample(size).cuda(0)

        # ------------alice------------
        generated_image, summed_codes = gen_one_img_ste(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=seed,
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
        save_file_stega = f'/groupshare_1/stegar/p={p}/{prompt_}_{seed}.png'
        write_bgr_img(generated_image, save_file_stega)
        # ------------alice------------

        # ------------attack------------
        # add_rayleigh_noise(save_file_stega, 0.1, save_file_stega)
        # add_salt_and_pepper_noise(save_file_stega, 0.005, 0.005, save_file_stega)
        # jpeg_attack(save_file_stega, save_file_stega)
        # ------------attack------------

        # ------------bob------------
        generate_img_raw = cv2.imread(save_file_stega)[:,:,::-1]

        # generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.01)

        generate_img = torch.Tensor(generate_img_raw.copy())
        generate_img = (generate_img.cuda(0) / 255 - 0.5) * 2
        # print(generate_img.shape)
        output = generate_img.permute(2,0,1).unsqueeze(0)
        rec = vae.forward_debug(output)

        # rec_img = (torch.clamp(rec[0][0], min=-1, max=1) /2 + 0.5) * 255
        # rec_img = rec_img.permute(1,2,0).cpu().numpy()
        z2 = rec[1]["h"].cuda(0)

        print(z2.shape)
        

        # with torch.amp.autocast('cuda:1', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        #     loss_function = torch.nn.MSELoss(reduction='sum')
        #     optimizer = torch.optim.Adam([z2], lr=0.1)
        #     lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
        #     decoder = vae.decoder.cpu().to('cuda:1')
        #     output2 = output.cuda(1)
        #     output2 = output2.detach()
        #     output2.requires_grad_(True)
        #     z2 = z2.detach()
        #     z2.requires_grad_(True)
        #     for _ in range(100):
        #         x_pred = decoder(z2)
        #         print('aaa')
        #         loss = loss_function(x_pred, output2)
        #         print('bbb')
        #         optimizer.zero_grad()
        #         print('ccc')
        #         loss.backward()
        #         print('ddd')
        #         optimizer.step()
        #         print('eee')
        #         lr_scheduler.step()

        # # print(rec[1])
        # z2 = z2.detach().cpu()
        # z2 = z2.cuda(0)
        decode_stega = z2.unsqueeze(-3) - summed_codes_list[-2]
        positives = torch.ones_like(decode_stega)
        negatives = torch.zeros_like(decode_stega)
        decode_stega_01 = torch.where(decode_stega > 0, positives, negatives).flatten().tolist()
        # print(len(decode_stega_01))
        plain_text = bob([int(b) for b in decode_stega_01], nonce, key)
        # print(plain_text)
        acc = compare_bit_acc(secret, plain_text)
        print(f'acc: {acc}')
        hs.append(acc)
        # stega_codes_01 = stega_codes / 0.1768
        # stega_err = (decode_stega_01 - stega_codes_01) /2

        # # print(stega_mask.flatten()[:30])
        # # print(stega_codes_01.shape)
        # print(summed_codes_list[-2].shape, summed_codes_list[-1].shape, z2.shape, decode_stega.shape, decode_stega_01.shape, stega_err.shape)
        # print((stega_err.abs()*stega_mask).flatten()[:30])
        # # print(decode_stega_01.flatten()[:30])
        # # print(max(stega_err.abs().flatten().cpu()))
        # print(f"z at p={p}:", (stega_err.abs()*stega_mask).sum())

        # # print(stega_codes_01.flatten()[:10])
        # # print(decode_stega_01.flatten()[:10])
        # print(f'rec at p={p}:', np.average(np.abs(generate_img_raw - rec_img)))
        # # print(f"z at p={p}:",np.average(np.abs((
        # #     (decode_stega_01 - stega_codes_01)*stega_mask
        # #     ).cpu())))

print(np.average(hs))