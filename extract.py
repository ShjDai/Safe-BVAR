import scipy.ndimage
from tools.run_infinity import *
import sys
import skimage
import scipy
import random
from PIL import Image, ImageFilter
from tqdm import trange
from water_common import data_num


if __name__ == '__main__':
    # import time
    # time.sleep(3600*4)
    # dataset_name = 'flickr'
    
    args = sys.argv
    image_type = args[1]
    dataset_name = args[2]
    atttack_type = args[3]
    # noise_type = args[2]
    # var = float(args[3])
    with open(f'/root/autodl-tmp/data/{dataset_name}/water256/random-bits.txt', 'r') as f1:
        hex_bits = f1.readlines()

    with open(f'/root/autodl-tmp/repo/mas_GRDH/text_prompt_dataset/{dataset_name}_dataset.txt', 'r') as f2:
        prompts = f2.readlines()


    def hex_to_bit_list(hex_str: str) -> list[int]:
        hex_str = hex_str.strip()
        int_list = [int(b, 16) for b in hex_str]
        bit_list = []
        for i in int_list:
            bit_list.extend([(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
        return bit_list


    # print(hex_bits[0], hex_to_bit_list(hex_bits[0])[:32])

    # exit()

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
    
    vae.eval()
    # cfg = 3
    # tau = 0.5
    # h_div_w = 1/1 # aspect ratio, height:width
    # seed = random.randint(0, 10000)
    # enable_positive_prompt=0

    # h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    # scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    # scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]


    hs = []
    # for idx, prompt in enumerate(tqdm(prompts)):
    for idx in trange(10):
    # for idx in trange(data_num[dataset_name]):
        # prompt = "The sun sets behind the mountains, painting the sky orange"
        # prompt_ = prompt
        # prompt_ = prompt_.replace(' ', '_')
        # common_kwargs["g_seed"] = idx
        # # for p in [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1, 0.5]:

        # # ------------bob------------
        # img, stages, summed_codes_list = gen_img_stages(infinity, vae, text_tokenizer, text_encoder, prompt, **common_kwargs)
        # base_save_file = f'/root/autodl-tmp/data/{dataset_name}/raw/{prompt_}.png'
        # write_bgr_img(img, base_save_file)

        # debug_codes = summed_codes_list[-1]
        # print(debug_codes.shape)

        # origin_bits = generate_random_bits(256)
        # with open('./test.txt', mode="a+") as f:
        # f.write(origin_bits.hex())
        prompt_ = prompts[idx]
        prompt_ = prompt_.replace(' ', '_')
        bits = hex_to_bit_list(hex_bits[idx])
        # ------------bob------------
        
        suffix = 'png' if image_type.startswith('p') else 'jpg'
        try:
            generate_img_raw = cv2.imread(f'/root/autodl-tmp/data/{dataset_name}/water256/{image_type}/{idx}.{suffix}')[:,:,::-1]
        except Exception as e:
            continue
        match atttack_type:
            case 'na':
                pass
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
            case 'g001':
                # img_shape = np.array(generate_img_raw).shape
                # g_noise = np.random.normal(0, 0.05, img_shape) * 255
                # g_noise = g_noise.astype(np.uint8)
                # generate_img_raw = np.clip(np.array(generate_img_raw) + g_noise, 0, 255)
                
                generate_img_raw = (np.array(generate_img_raw).astype(np.uint8) / 255 - 0.5) * 2
                generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.01)
                # generate_img_raw = np.array(generate_img_raw * 255, dtype=np.uint8)
                generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
            case 'gb2':
                # img = Image.fromarray(generate_img_raw)
                # img = img.filter(ImageFilter.GaussianBlur(radius=2))
                # generate_img_raw = np.array(img)
                # generate_img_raw = cv2.GaussianBlur(generate_img_raw, (3,3), 1, sigmaY=1)
                generate_img_raw = (generate_img_raw / 255 - 0.5) * 2
                # generate_img_raw = generate_img_raw / 255
                # generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='s&p', amount=0.005)
                generate_img_raw = skimage.filters.gaussian(generate_img_raw, sigma=2)
                # print(generate_img_raw)
                # generate_img_raw = scipy.ndimage.uniform_filter(generate_img_raw, 3)
                generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255)
                
                # img = Image.fromarray(generate_img_raw)
                # generate_img_raw = img.filter(ImageFilter.GaussianBlur(radius=5))
                # generate_img_raw = np.array(generate_img_raw)
                # print(generate_img_raw)
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
            case 'g003':
                generate_img_raw = (np.array(generate_img_raw).astype(np.uint8) / 255 - 0.5) * 2
                generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.0225)
                generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
            case 'rs4':
                rss = 256
                generate_img_raw = cv2.resize(generate_img_raw, (rss,rss))
                generate_img_raw = cv2.resize(generate_img_raw, (1024,1024))
        
        # print(generate_img_raw)
        # generate_img_raw = (generate_img_raw / 255 - 0.5) * 2
        # generate_img_raw = skimage.filters
        # generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='s&p', amount=0.005)
        # generate_img_raw = skimage.filters.gaussian(generate_img_raw, sigma=1)
        # generate_img_raw = scipy.ndimage.uniform_filter(generate_img_raw, 3)
        # generate_img_raw = np.array((generate_img_raw / 2 + 0.5) * 255, dtype=np.uint8)
        
        # rss = 512
        # generate_img_raw = cv2.resize(generate_img_raw, (rss,rss))
        # generate_img_raw = cv2.resize(generate_img_raw, (1024,1024))
        
        # print(generate_img_raw)

        generate_img = torch.Tensor(generate_img_raw.copy()).cuda()
        generate_img = (generate_img / 255 - 0.5) * 2
        # # print(generate_img.shape)
        output = generate_img.permute(2,0,1).unsqueeze(0)
        # rec = vae.forward_debug(output)
        

        # # rec_img = (torch.clamp(rec[0][0], min=-1, max=1) /2 + 0.5) * 255
        # # rec_img = rec_img.permute(1,2,0).cpu().numpy()
        # z2 = rec[1]["h"].cuda(0)
        
        z2 = vae.encode_for_extract(output)

        # print(z2.shape)
        # i = 16
        # j = 16
        # recover_bits = []
        # for i_ in range(16):
        #     for j_ in range(16):
        #         i = i_ * 4 + 2
        #         j = j_ * 4 + 2
        #         NUM = 32
        #         es = [e[i][j] for e in z2[0][:NUM]]
        #         # avg = sum(es)/len(es)
        #         avg = torch.Tensor(es).mean().item()
        #         # print('0,0:', avg)
        #         es2 = [e[i][j+1] for e in z2[0][:NUM]]
        #         # avg2 = sum(es2)/len(es2)
        #         avg2 = torch.Tensor(es2).mean().item()
        #         # print('0,1:', avg2)
        #         es3 = [e[i+1][j] for e in z2[0][:NUM]]
        #         # avg3 = sum(es3)/len(es3)
        #         avg3 = torch.Tensor(es3).mean().item()
        #         # print('1,0:', avg3)
        #         es4 = [e[i+1][j+1] for e in z2[0][:NUM]]
        #         # avg4 = sum(es4)/len(es4)
        #         avg4 = torch.Tensor(es4).mean().item()
        #         # print('1,1:', avg4)
        #         es5 = [e[i-1][j] for e in z2[0][:NUM]]
        #         avg5 = sum(es5)/len(es5)
        #         # print('-1,0:', avg5)
        #         es6 = [e[i-1][j+1] for e in z2[0][:NUM]]
        #         avg6 = sum(es6)/len(es6)
        #         # # print('-1,1:', avg6)
        #         es7 = [e[i][j-1] for e in z2[0][:NUM]]
        #         avg7 = sum(es7)/len(es7)
        #         # # print('0,-1:', avg7)
        #         es8 = [e[i+1][j-1] for e in z2[0][:NUM]]
        #         avg8 = sum(es8)/len(es8)
        #         # # print('1,-1:', avg8)
        #         es9 = [e[i-1][j-1] for e in z2[0][:NUM]]
        #         avg9 = sum(es9)/len(es9)
        #         avg_avg = sum([avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9]) / 8
        #         # avg_avg = sum([avg2, avg3, avg4]) / 3
        #         if avg_avg > avg:
        #             recover_bits.append(0)
        #         else:
        #             recover_bits.append(1)
        
        recover_bits = []
        # for i in range(16):
            # for j in range(16):
                # i = i_ * 4 + 2
                # j = j_ * 4 + 2
        for bit_idx in range(256):
                # bit_idx = i * 16 + j
            
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

        err_num = 0
        for b1, b2 in zip(recover_bits, bits):
            if b1 != b2:
                err_num += 1
        hs.append(err_num / 256)

    print(1- np.average(hs))
