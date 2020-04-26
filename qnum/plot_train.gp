set xlabel 'Samples'
set ylabel '?'
#set logscale y
#set parametric
plot 'train.unigram.stats.tsv' using 1:3:2 
