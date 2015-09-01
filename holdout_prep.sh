cat sampleSubmission.csv | cut -d, -f1 > test.csv
cat train.csv | cut -d, -f1 > valid.csv
tail -n +2 train.csv | gshuf > train_shuffled.csv
head -n 25000 train_shuffled.csv > holdout.csv
tail -n +25001 train_shuffled.csv > train_no_holdout.csv