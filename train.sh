#!/bin/bash
source venv/bin/activate
echo "********************************* train funsd ****************************************"
python train.py train -d FUNSD -hs 64 -hl 128
mv file.log data/file_FUNSD_GNN.log

echo "********************************* train xfund ****************************************"
python train.py train -d XFUNSD -hs 64 -hl 128
mv file.log data/file_XFUNSD_GNN.log