# PIDM - Person Image Synthesis via Denoising Diffusion Model

 <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2104.0ede">ArXiv</a>
    | 
    <a href="">Paper</a>
  </b>
</p> 

<p align="center">
<img src=Figures/github.jpg>
</p>

 
[Ankan Kumar Bhunia](https://scholar.google.com/citations?user=2leAc3AAAAAJ&hl=en),
[Salman Khan](https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en),
[Hisham Cholakkal](https://scholar.google.com/citations?user=bZ3YBRcAAAAJ&hl=en), 
[Rao Anwer](https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en),
[Jorma Laaksonen](https://scholar.google.com/citations?user=qQP6WXIAAAAJ&hl=en),
[Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en) &
[Fahad Khan](https://scholar.google.ch/citations?user=zvaeYnUAAAAJ&hl=en&oi=ao)

> **Abstract:** 
The pose-guided person image generation task requires
synthesizing photorealistic images of humans in arbitrary
poses. The existing approaches use generative adversar-
ial networks that do not necessarily maintain realistic tex-
tures or need dense correspondences that struggle to han-
dle complex deformations and severe occlusions. In this
work, we show how denoising diffusion models can be ap-
plied for high-fidelity person image synthesis with strong
sample diversity and enhanced mode coverage of the learnt
data distribution. Our proposed Person Image Diffusion
Model (PIDM) disintegrates the complex transfer problem
into a series of simpler forward-backward denoising steps.
This helps in learning plausible source to target transfor-
mation trajectories that result in faithful textures and undis-
torted appearance details. We introduce a ‘texture diffusion
module’ based on cross-attention to accurately model the
correspondences between appearance and pose informa-
tion available in source and target images. Further, we pro-
pose ‘disentangled classifier-free guidance’ to ensure close
resemblance between the conditional inputs and the synthe-
sized output in terms of both pose and appearance informa-
tion. Our extensive results on two large-scale benchmarks
and a user study demonstrate the photorealism of our pro-
posed approach under challenging scenarios. We also show
how our generated images can help in downstream tasks
