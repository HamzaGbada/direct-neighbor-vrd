#!/bin/bash
source venv/bin/activate
echo "********************************* train embed funsd ****************************************"
#python train_word_embedding.py embed -d FUNSD -e 25

mv file.log data/file_FUNSD_embedding.log
echo "********************************* train embed xfund ****************************************"
#python train_word_embedding.py embed -d XFUND -e 25

#mv file.log data/file_XFUND_embedding.log

#python builder.py build -d XFUND
python builder.py build -d FUNSD