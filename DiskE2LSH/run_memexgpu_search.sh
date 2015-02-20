CODE_PATH=/home/rgirdhar/data/Work/Code/0002_Retrieval/ScalableLSH/DiskE2LSH
$CODE_PATH/main.bin \
    -d /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_pool5_uni_normed_LMDB \
    -n /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt \
    -c /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/selsearch_boxes/counts.txt \
    -l /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/search_500_225bit.index \
    -o /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches/ \
    -m /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/TestList.txt
#    -q /home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/features/CNN_pool5_copy \
