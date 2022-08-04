# Thesis project: Improving genetic association using deep learning
![alt text](https://github.com/neggor/improving-genetic-association-using-deep-learning/blob/main/images/Other/Introductory_plot.png)

Raw data is available in a private Kaggle repository. Access may be requested.

All results are available here.

Reproducibility usage:

- All representations (traits) generated through Deep Learning algorithms can be extracted by running 'run_deep_models.py'. It will download the raw data and do all the cleaning processes and feature engineering. Then it will fit all the models and extract the representations.

- With the representations: GWAS, heritability and other statistics used can be replicated by runing 'GWAS_R2_H2.R' in src/model/.

- With representations and statistics: All generated images can be replicated using 'Plot_maker.ipynb' and 'Plot_maker.Rmd'.
