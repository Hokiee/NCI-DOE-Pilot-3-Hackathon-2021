# NCI-Hackathon-2021


### Data: TCGA
- Copy engress_TCGA/features_full into data/features_full
- Copy engress_TCGA/histo_meta.csv to data/histo_meta.csv

### Environment: 
 [Summit-Ascent](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#training-system-ascent)  
 [Module instructions](https://docs.olcf.ornl.gov/software/analytics/ibm-wml-ce.html) (ibm-wml-ce/1.6.2-5) or (open-ce for HiBert)

### Code: 

#### Train Test Split : Creates files under data/split
    python trainTestSplitMetaData.py
	
#### Data Handler : Creates files under data/npy , data/mapper, data/word2idx.pkl and data/vocab.npy
    python data_handler.py

### Models          
##### 1. MTCNN Hard Paramenter Sharing ( TF-1 ) 
    cd mtcnn
    python mt_cnn_exp.py 
    OR
    cd mtcnn
    bsub mt_cnn_exp.lsf
##### 2. MTCNN Cross stitch (TF-1)
    cd mtcnn
    python tf_mtcnn_cs.py
    OR
    cd mtcnn
    bsub tf_mtcnn_cs.lsf
##### 3. HiSAN ( TF-1 )
    cd Hisan
    python tf_mt_hisan.py
    OR
    cd Hisan
    bsub hisan.lsf
##### 4. HiBERT ( Pytorch)
    cd HiBert 
    python huggingface_pool_multigpu.py
    OR
    cd HiBert
    bsub horovod.lsf
##### 5. LANL Abstaining Code MTCNN
##### 6. LANL Abstaining Code MTHiSAN

