#!/bin/bash

logfile=$1
expname=${logfile%-log.*}
echo "Splitting experiment log for $expname"
echo "i,smp,loss,acc" > "$expname-train.csv"
fgrep '[TRAIN]' "$logfile" | sed 's/\[TRAIN\] //g' | sed 's/\s*\w\w*\s*\=\s*//g' | sed  '/^epoch.*$/d' | awk '{print NR "," $s}' >> "$expname-train.csv"
fgrep '[DEBUG]' "$logfile" | sed 's/\[DEBUG\] //g' > "$expname-debug.txt"
echo "epoch,loss,acc" > "$expname-test.csv"
fgrep '[TEST]' "$logfile"  | sed 's/\[TEST\] //g' | paste -sd ',\n' | sed 's/[^0-9.,]//g' >> "$expname-test.csv"
