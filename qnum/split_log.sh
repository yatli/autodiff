#!/bin/bash

logfile=$1
expname=${logfile%.log}
echo "Splitting experiment log for $expname"
echo -e "num_updates\tepoch\tbatch_loss\tavg_loss\tacc" > "$expname-train.tsv"
# sample line:
#[TRAIN] epoch=   0 step=    40 batchloss=      2.30156 avgloss=      2.30449 acc=       0.1625 nupdates=         80
# 1        2      3  4       5  6               7        8             9       10           11    12              13
fgrep '[TRAIN]' "$logfile" | awk '{OFS="\t"} {print $13, $3, $7, $9, $11}' >> "$expname-train.tsv"
fgrep '[DEBUG]' "$logfile" | sed 's/\[DEBUG\] //g' > "$expname-debug.txt"
#sample line:
# [TEST] epoch    0 avgloss=     0.153012 acc=       0.9532
# 1       2       3  4              5     6              7
echo -e "epoch\tloss\tacc" > "$expname-test.tsv"
fgrep '[TEST]' "$logfile"  | awk '{OFS="\t"} {print $3, $5, $7}' >> "$expname-test.tsv"
