# Safe-BVAR

paper link: [Safe-BVAR](https://dl.acm.org/doi/abs/10.1145/3746027.3755138)

## Abstract

Bitwise Vision AutoRegressive (BVAR) Model, as a distinguished source of young blood, has been taking the lead in the track of text-to-image synthesis, which at the same time raises legal and ethnic concerns such as copyright and authenticity. However, existing methods mainly focus on watermarking within diffusion models, which rely on the distinctive attributes of diffusion steps and cannot be directly transferred to new circumstances. To this end, we propose Safe-BVAR, the first watermark framework to embed bit strings during image generation in BVAR. Our study discovers the local similarity of the inferenced latent feature and the element-wise robustness of image autoencoder. Therefore, combined with the residual-accumulative nature of BVAR, we propose a novel Late Stage Residual Implanter to embed watermark and extract the information based on Local Contextual Extractor. Furthermore, we propose a Distributed Rotational Arranger to enhance watermark against local distortions. Our method is training-free and plug-and-play. Meanwhile, it can be easily applied to flexible-sized images. We evaluate the robustness and invisibility of the watermark, showing that it can resist common image attacks and cast inappreciable influence on the image.

## TODOs

1. upload watermark codes
2. a better interface


## Thanks

We utilize codes and weights from [Infinity](https://github.com/FoundationVision/Infinity) and prompts from [GRDH](https://github.com/FoundationVision/Infinity). Thanks to them!