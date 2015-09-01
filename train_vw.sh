#! /bin/sh
DATE=`date +%Y%m%d`
modelname="$DATE-$1.vw"
shift

set -x

vw -d intermediate/vw_labelled.txt -c --loss_function logistic -f models/$modelname $@