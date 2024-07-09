## MERL
[Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement](https://arxiv.org/abs/2403.06659), ICML 2024.

![framework](docs/framework.png)

###  Installation
To clone this repository:
```
git clone https://github.com/cheliu-computation/MERL.git
```

### Dataset downloading
Datasets we used are as follows:
- **MIMIC-IV-ECG**: We downloaded the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset as the ECG signals and paired ECG reports.

- **PTB-XL**: We downloaded the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset which consisting four subsets, Superclass, Subclass, Form, Rhythm.

- **CPSC2018**: We downloaded the [CPSC2018](http://2018.icbeb.org/Challenge.html) dataset which consisting three training sets. 

- **CSN(Chapman-Shaoxing-Ningbo)**: We downloaded the [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/) dataset.


### Data Preprocessing
We preprocessed pretraining datasets and split the dataset into train/val set using the code in `pretrain/preprocess.ipynb`.\
We preprocessed downstream datasets and split the dataset into train/val/test set using the code in `finetune/preprocess.ipynb`.\
We also provide the train/val/test split csv file in `finetune/data_split`

### Pre-training

We pre-trained MERL on MIMIC-IV-ECG using this command:

```
bash MERL/pretrain/launch.sh
```

Pre-trained models can be found [here](https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2?usp=drive_link).\
We uploaded the pretrained models with resenet and vit.\
xxx_ckpt.pth is the whole pretrained model for zeroshot classification.\
xxx_encoder.pth is the ecg encoder only for linear probing.

### Downstream tasks
We evlauate the performance of MERL on three scenarios: zero-shot classification, linear probing, and domain transferring.

#### zero-shot classification
We evaluate linear classification performance of our model using this command:
```
cd MERL/zeroshot
bash zeroshot.sh
```
We also release the CKEPE prompt in `zeroshot/CKEPE_prompt.json`.\
Due to the copyright, we are unable to release the original SCP-code database, but you can find all information in: [https://www.iso.org/standard/84664.html](https://www.iso.org/standard/84664.html).

#### linear probing
We provide bash script for evaluating linear probing performance of MERL:
```
cd MERL/finetune/sub_script
bash run_all_linear.sh
```
You can use `--dataset` to set specific dataset for finetuning. Here, 3 datsets are available: chexpert, rsna and covidx.
You can use `--ratio` to set the fraction of training data for finetuning.

#### domain transferring
For domain trasnfering scenario, you do not reimplement any new experiments. You can only compute the metric across the overlapped categories.

### Reference
If you found our work useful in your research, please consider citing our works(s) at:
```bash
@inproceedings{liuzero,
  title={Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement},
  author={Liu, Che and Wan, Zhongwei and Ouyang, Cheng and Shah, Anand and Bai, Wenjia and Arcucci, Rossella},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
