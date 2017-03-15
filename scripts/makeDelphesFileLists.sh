#!/bin/bash

#dir=/global/cscratch1/sd/wbhimji/DelphesOutput/PU-HighRes-2

# No pileup samples
dir=/global/cscratch1/sd/wbhimji/DelphesOutput/NoPU
# RPV
ls $dir/RPV10_1400_850*.root > data/delphes_noPU_rpv_1400_850.txt
# QCD background samples
jzlist="3 4 5 6 7 8 9 10 11 12"
for jzi in $jzlist; do
    fileList=data/delphes_noPU_qcd_JZ${jzi}.txt
    ls $dir/QCDBkg_JZ${jzi}*.root > $fileList
done
