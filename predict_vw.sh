#! /bin/bash
modelfile=$1
modelname=`basename $modelfile`
predfile="predictions/$modelname.pred.txt"

set -x

vw -d intermediate/vw.txt -c -t -i $modelfile -p $predfile --loss_function logistic