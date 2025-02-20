# RAS

Official implementation of [Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection](https://ebooks.iospress.nl/doi/10.3233/FAIA240728), accepted by ECAI 2024.
![pipeline](https://github.com/Nothingtolose9979/RAS/blob/main/figures/pipeline.png)

## Preparation

### Datasets

Please download the [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset, [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) dataset, [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip) dataset and [MPDD](https://github.com/stepanje/MPDD) dataset, and put them at `./data`.

```
|-- data
	|-- btad
		|-- 01
		|-- 02
		|-- 03
	|-- mpdd
		|-- bracket_black
		|-- bracket_brown
		...
	|-- mvtec
		|-- bottle
		|-- cable
		...
	|-- visa
		|-- 1cls
			|-- candle
			|-- capsules
			...
```



### Environment

```bash
conda create -n ras python=3.8
conda activate ras
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```



## Training & Evaluation

**train**

```bash
sh ras_train.sh
```

**eval**

```bash
sh ras_eval.sh
```



## Acknowledgement

We acknowledge the excellent implementation from [UniAD](https://github.com/zhiyuanyou/UniAD/tree/main).



## Citation

If our code or models help your work, please cite our paper:

```latex
@incollection{yang2024context,
  title={Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection},
  author={Yang, Hui-Yue and Chen, Hui and Liu, Lihao and Lin, Zijia and Chen, Kai and Wang, Liejun and Han, Jungong and Ding, Guiguang},
  booktitle={ECAI 2024},
  pages={2098--2105},
  year={2024},
  publisher={IOS Press}
}
```
