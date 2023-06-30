# Food Recognition

# Dataset

* [sg-food-233](https://drive.google.com/file/d/15jF8sDkdr7TRrV5xhG1QAq2lm8qhdcn2/view?usp=sharing)

## Experiments

| Dataset | Neural Network | Accuracy | Accuracy Top 5 |
| :----: | :----: | :----: | :----: |
| sg-food-233 | [VIT-B32](https://drive.google.com/file/d/158QRNobdUAq81yKnf-DQUurDXS7X7Cu8/view?usp=share_link) | 77.30% | 93.39% |
| sg-food-233 | [Xception](https://drive.google.com/file/d/1W75SAHYP7zhubiU4QzTPhqQuC_CvZWjJ/view?usp=share_link) | 69.08% | 89.24% |
| sg-food-233 | [DenseNet121](https://drive.google.com/file/d/1-7GiASFCHFlM_iS9WDDtP1SuJAV-Dbw6/view?usp=share_link) | 68.70% | 88.60% |
| sg-food-233 | [DenseNet201](https://drive.google.com/file/d/1fFNB8SYGkWA-0j9Y2jcgBec7VNn1j5KV/view?usp=share_link) | 70.30% | 89.49% |
| sg-food-233 | [ResNet152V2](https://drive.google.com/file/d/1x2P6RFlWQPuJy0ha4CUxCsSr3-oemHFh/view?usp=share_link) | 70.40% | 89.80% |
| sg-food-233 | [InceptionV3](https://drive.google.com/file/d/1tG9k3ih9np5_TRPUn81WaImBS7bKbQwL/view?usp=share_link) | 63.19% | 86.01% |
| sg-food-233 | [InceptionResNetV2](https://drive.google.com/file/d/1k0ZP7eAqm2dH-FdBuAncyFGAFg2jYRHD/view?usp=share_link) | 68.99% | 89.54% |
| tw-food-101 | [Xception](https://drive.google.com/file/d/1ekOl6HT8jjl2FQJ2SvZEw06XGyhXlPj9/view?usp=share_link) | 79.41% | 95.27% |
| tw-food-101 | [DenseNet201](https://drive.google.com/file/d/1MRywupyObsFS5J_KTQEFUzoFGKwsfgzV/view?usp=share_link) | 80.61% | 94.53% |
| food-101 | [DenseNet201](https://drive.google.com/file/d/1FKUluEpOQE4Vk32JoreB5O8c9jH9Tm92/view?usp=share_link) | 70.99% | 90.34% |

## Deploy Docker

```
docker build --no-cache -t food-recognition:v1.0 .
docker save -o food-recognition-v1.0.tar food-recognition:v1.0
docker load -i food-recognition-v1.0.tar
docker images
docker run -p 5050:5050 food-recognition:v1.0
```

## Categories need to remove or not exist in empower dataset

* CNY love letter
* Sweets
* bakso
* begedil
* chawanmushi
* pitaya
* tumpeng
* buckwheat

## Todo

* InceptionV3 log and figure need to download
