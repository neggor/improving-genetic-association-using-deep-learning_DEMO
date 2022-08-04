import subprocess
import os

## SET UP (Download data, clean it, split it)
subprocess.run('python src/data/set_up.py -dr 1 -dc 0 -pca 0 -snp 86001', shell = True) 


## SUPERVISED
# NO GENOTYPE STRATIFICATION:
## INCEPTION TIME:
### All features no downsampling
subprocess.run('python src/model/inception_time.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 \
    -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 \
        -c 18 -c 19 -c 20  -snp 86001 -gs 0 -bs 20' , shell = True)

### Just speed donwsampling by 10
subprocess.run('python src/model/inception_time.py -d 10 -c 9 -snp 86001 -gs 0 -bs 32' , shell = True)

## Simple Transformer
subprocess.run('python src/model/transformer.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 -c 5 -c 6 -c 7\
     -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 -c 18 -c 19 -c 20\
           -snp 86001 -gs 0 -bs 32' , shell = True)
## XCEPTION
subprocess.run('python ./src/model/Xception.py -gs 0 -snp 86001 -res 128 -bs 32' , shell = True)


# GENOTYPE STRATIFICATION:
### All features no downsampling
subprocess.run('python src/model/inception_time.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 \
    -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 \
        -c 18 -c 19 -c 20  -snp 86001 -gs 1 -bs 20' , shell = True)

### Just speed donwsampling by 10
subprocess.run('python src/model/inception_time.py -d 10 -c 9 -snp 86001 -gs 1 -bs 32' , shell = True)

## Simple Transformer
subprocess.run('python src/model/transformer.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 -c 5 -c 6 -c 7\
     -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 -c 18 -c 19 -c 20\
           -snp 86001 -gs 1 -bs 32' , shell = True)
## XCEPTION
subprocess.run('python ./src/model/Xception.py -gs 1 -snp 86001 -res 128 -bs 32' , shell = True)

# ANTI-GENOTYPE STRATIFICATION:
### All features no downsampling
subprocess.run('python src/model/inception_time.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 \
    -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 \
        -c 18 -c 19 -c 20  -snp 86001 -gs 2 -bs 20' , shell = True)

### Just speed donwsampling by 10
subprocess.run('python src/model/inception_time.py -d 10 -c 9 -snp 86001 -gs 2 -bs 32' , shell = True)

## Simple Transformer
subprocess.run('python src/model/transformer.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 -c 18 -c 19 -c 20 -snp 86001 -gs 2 -bs 32' , shell = True)
## XCEPTION
subprocess.run('python ./src/model/Xception.py -gs 2 -snp 86001 -res 128 -bs 32' , shell = True)

#=============================================================================
## SELF-SUPERVISED
subprocess.run('python src/model/minimal_TCN_VAE.py -d 1 -c 0 -c 1 -c 2 -c 3 -c 4 -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 -c 18 -c 19 -c 20  -gs 1 -bs 32', shell = True)
#============================================================================
## CONTRASTIVE
subprocess.run(' python src/model/minimal_TCN_Contrastive.py -d 1 -c 1 -c 2 -c 3 -c 4 -c 5 -c 6 -c 7 -c 8 -c 9 -c 10 -c 11 -c 12 -c 13 -c 14 -c 15 -c 16 -c 17 -c 18 -c 19 -c 20 -bs 128', shell = True)

