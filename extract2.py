from tools.run_infinity import *
import sys
from tqdm import trange


if __name__ == '__main__':
    import time
    # time.sleep(14500)
    
    args = sys.argv
    attack_type = args[1]
    with open('/root/autodl-tmp/data/ms-coco/water256/random-bits.txt', 'r') as f1:
        hex_bits = f1.readlines()

    with open('/root/autodl-tmp/repo/mas_GRDH/text_prompt_dataset/coco_dataset.txt', 'r') as f2:
        prompts = f2.readlines()


    def hex_to_bit_list(hex_str: str) -> list[int]:
        hex_str = hex_str.strip()
        int_list = [int(b, 16) for b in hex_str]
        bit_list = []
        for i in int_list:
            bit_list.extend([(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
        return bit_list

    dataset_name = 'ms-coco'
    vae_path='./weights/infinity_vae_d32_reg.pth'
    args=argparse.Namespace(
        pn='1M',
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
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        bf16=1
    )

    # load vae
    print('load vae...')
    vae = load_visual_tokenizer(args)
    
    vae.eval()
    hs = []
    # for idx, prompt in enumerate(tqdm(prompts)):
    for idx in trange(5000):
        bits = hex_to_bit_list(hex_bits[idx])
        # ------------bob------------
        generate_img_raw = cv2.imread(f'/root/autodl-tmp/data/ms-coco/water256/{attack_type}/{idx}.jpg')[:,:,::-1]
        # # generate_img_raw = skimage.util.random_noise(generate_img_raw, mode='gaussian', var=0.01)

        generate_img = torch.Tensor(generate_img_raw.copy())
        generate_img = (generate_img.cuda(0) / 255 - 0.5) * 2
        output = generate_img.permute(2,0,1).unsqueeze(0)
        rec = vae.forward_debug(output)
        z2 = rec[1]["h"].cuda(0)

        recover_bits = []
        for i in range(16):
            for j in range(16):
                bit_idx = i * 16 + j
                NUM = 32
                es = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
                avg = sum(es)/len(es)
                es2 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
                avg2 = sum(es2)/len(es2)
                es3 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
                avg3 = sum(es3)/len(es3)
                es4 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
                avg4 = sum(es4)/len(es4)
                es5 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 2)%64] for c in range(NUM)]
                avg5 = sum(es5)/len(es5)
                es6 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 3)%64] for c in range(NUM)]
                avg6 = sum(es6)/len(es6)
                es7 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 2)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
                avg7 = sum(es7)/len(es7)
                es8 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 3)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
                avg8 = sum(es8)/len(es8)
                es9 = [z2[0][c][((bit_idx - c*8)//16 * 4 + 1)%64][((bit_idx - c*8)%16 * 4 + 1)%64] for c in range(NUM)]
                avg9 = sum(es9)/len(es9)
                avg_avg = sum([avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9]) / 8
                # avg_avg = sum([avg2, avg3, avg4]) / 3
                recover_bits.append(0 if avg_avg>avg else 1)
        err_num = 0
        for b1, b2 in zip(recover_bits, bits):
            if b1 != b2:
                err_num += 1
        hs.append(err_num / 256)
    print(1- np.average(hs))
