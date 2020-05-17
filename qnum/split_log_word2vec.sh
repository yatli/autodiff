#!/bin/bash

logfile=$1
expname=${logfile%.log}
echo "Splitting experiment log for $expname"
echo -e "${expname}-avg-loss" > "$expname.tsv"
# sample line:
# Progress:   0.0% words/sec/thread:  125212 lr:  0.024990 avg.loss:  4.144543 ETA:   0h 4m 8s
# 1            2    3                  4       5    6        7          8        9     10 11 12
cat "$logfile" | awk '{OFS="\t"} {print $8}' >> "$expname.tsv"
