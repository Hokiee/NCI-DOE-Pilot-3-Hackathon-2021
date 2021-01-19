# NCI-Hackathon-2021

### Environment: 
 [Summit-Ascent](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#training-system-ascent)  

To login to Ascent and clone project:
```
ssh -o ServerAliveInterval=60 <ucams>@login1.ascent.olcf.ornl.gov (use ucams password)
cd /gpfs/wolf/proj-shared/gen149/
mkdir <ucams>
cd <ucams>
git clone https://code.ornl.gov/93t/nci-hackathon-2021.git
```

### Setting up LSF scripts:

In the line `#BSUB -m "login1 h49nXX"`, replace `XX` with your two digit team number from the table below.

| Team Institution(s) | Team Number |
| ----------- | ----------- |
| U Utah | 01 |
| U Wisconsin | 02 |
| NCI | 03 |
| GenomOncology | 04 |
| U Michigan | 05 |
| EKU | 06 |
| UNC | 07 |
| U Colorado | 08 |
| U Arkansas | 09 |
| NMCSD & DOD | 10 |
| Cancer Data Registry of Idaho | 11 |
| U Kentucky | 12 |
| Med U South Carolina | 13 |
| CDC | 14 |
| Rutgers & NJIT | 15 |
| City of Hope | 16 | 

To launch an interactive shell:
```
bsub -Is -W 1:00 -nnodes 1 -P gen149 $SHELL
module load ibm-wml-ce/1.6.2-5
ENVROOT=/gpfs/wolf/proj-shared/gen149/j8g
conda activate $ENVROOT/ibmwmlce
export PATH=$ENVROOT/ibmwmlce/bin:$PATH
jsrun -n1 -c7 -g6 -r1 hostname
```

### Data Prepro: 
```
cd /gpfs/wolf/proj-shared/gen149/<ucams>/nci-hackathon-2021
cp -r /gpfs/wolf/proj-shared/gen149/data .
```
#### Create data inputs for all models except BERT:
    python trainTestSplitMetaData.py
    python data_handler.py
    OR
    bsub data_setup_tf.lsf
	
#### Create data inputs for BERT:
    cd HiBERT
    python huggingface_dataloader.py
    OR
    bsub data_setup_bert.lsf
##### Interacting with Jobs : https://docs.olcf.ornl.gov/systems/summit_user_guide.html#interacting-with-jobs

### Models          
##### 1. MTCNN Hard Parameter Sharing (TF1) 
    cd mtcnn
    python mt_cnn_exp.py 
    OR
    cd mtcnn
    bsub mt_cnn_exp.lsf
##### 2. MTCNN Cross stitch (TF1)
    cd mtcnn
    python tf_mtcnn_cs.py
    OR
    cd mtcnn
    bsub tf_mtcnn_cs.lsf
##### 3. HiSAN (TF1)
    cd Hisan
    python tf_mthisan_new.py
    OR
    cd Hisan
    bsub hisan.lsf
##### 4. HiBERT (Pytorch)
    cd HiBert 
    python huggingface_pool_multigpu.py
    OR
    cd HiBert
    bsub horovod.lsf
##### 5. LANL Abstaining Code MTCNN
    cd mtcnn
    python abs_mt_cnn_exp.py 
    OR
    cd mtcnn
    bsub abs_mt_cnn_exp.lsf
