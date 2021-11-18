# Interpretable-NLP-Transformers
CS 4550 Semester Project


To run classification task:

1) Retrieve the checkpoint data
2) Unzip into the bert_base directory (the only missing file is the file ending with ckpt..data0000-0001)
3) Execute in terminal:\
Windows version
python run_classifier.py --task_name=cola --vocab_file=".\bert_base\vocab.txt\" \\\
--bert_config_file=".\bert_base\bert_config.json" --output_dir=".\output" \\\
--init_checkpoint=".\bert_base\bert_model.ckpt" --data_dir=".\SentimentalLIAR" \\\
--do_train=True --do_eval=True\
Linux Version:\
python3 run_classifier.py --task_name=cola --vocab_file="./bert_base/vocab.txt" \\\
--bert_config_file="./bert_base/bert_config.json" --output_dir="./output" \\\
--init_checkpoint="./bert_base/bert_model.ckpt" --data_dir="./SentimentalLIAR" \\\
--do_train=True --do_eval=True\
Hyperparameter Tuning:\
python3 "./hyperparameter tuning.py"
NOTE: The flags tell the script where the input is, what format it is in, where to put the output, what pre-trained 
information to start with, and to run fine-tuning.

# TODO
1) Determine the optimal number of epochs to useDetermine the optimal learning rate
2) Determine how to pull the information about the attention heads
3) Compare the untrained, unoptimized results to our fine-tuned model to draw some conclusion about how BERT learns.