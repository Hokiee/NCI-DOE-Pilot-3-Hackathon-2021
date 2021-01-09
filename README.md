# NCI-Hackathon-2021

### Environment: 
 [Summit-Ascent](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#training-system-ascent)  

To login to Ascent and clone project:
```
ssh <ucams>@login1.ascent.olcf.ornl.gov (use ucams password)
cd /gpfs/wolf/proj-shared/gen149/
mkdir <ucams>
cd <ucams>
git clone https://code.ornl.gov/93t/nci-hackathon-2021.git
```

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

#### Create data inputs for all models except BERT:
    python trainTestSplitMetaData.py
    python data_handler.py
    OR
    bsub data_setup_tf.lsf
	
#### Create data inputs for BERT:
    cd HiBERT
    python huggingface_dataloader.py
    OR
    busb data_setup_bert.lsf

### Models          
##### 1. MTCNN Hard Paramenter Sharing (TF1) 
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

