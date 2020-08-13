## PixelCNN++ image modelling
Trains a PixelCNN++ model [(Salimans et al.,
2017)](https://arxiv.org/abs/1701.05517) for image generation on the CIFAR-10 dataset.
Only unconditional image generation is implemented, trained using ADAM
on the negative log-likelihood. As in the original [OpenAI implementation](https://github.com/openai/pixel-cnn)
we use weightnorm parameterization with data-dependent initialization.

Code for sampling is also provided. The following image, containing 256 samples, was generated in 4m 24s
on an 8 x Nvidia V100 machine.

![alt text](sample.png "PixelCNN++ samples.")
### Requirements (Training)
* [TF datasets](https://www.tensorflow.org/datasets), which will download and cache the CIFAR-10 dataset the first time you
  run `train.py`.

### Requirements (Sampling)
* [Pillow](https://pillow.readthedocs.io/en/stable/) for saving samples as PNG files.

### Supported setups
The model should run with other configurations and hardware, but was tested on the following.

| Hardware | Batch size | Training time | Log-likelihood (bits/dimension) | TensorBoard.dev |
| --- | --- | --- | --- | --- |
| 8 x TPUv3  | 256  |   8h 55m | 3.03 | [2020-08-13](https://tensorboard.dev/experiment/XpXXKlOxRjmu8P2z49rPAA/) |

### How to run
#### 8 x TPUv3
To run training
```
python train.py --batch_size=320
```
To run sampling (this will automatically load model parameters from the most recent trained checkpoint)
```
python sample.py --sample_batch_size=256
```
