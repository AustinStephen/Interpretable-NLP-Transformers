# Interpretable-NLP-Transformers
CS 4550 Semester Project


To run classification task:

1) Retrieve the checkpoint data
2) Unzip into the bert_base directory (the only missing file is the file ending with ckpt..data0000-0001)
3) Execute in terminal:
python run_classifier.py --task_name=cola --vocab_file=".\bert_base\vocab.txt\" \\\
--bert_config_file=".\bert_base\bert_config.json" --output_dir=".\output" \\\
--init_checkpoint=".\bert_base\bert_model.ckpt" --data_dir=".\SentimentalLIAR" \\\
--do_train=True --do_eval=True\
NOTE: The flags tell the script where the input is, what format it is in, where to put the output, what pre-trained 
information to start with, and to run fine-tuning.

#TODO
1) Determine how to extract the performance after each epoc
2) Determine the optimal number of epochs to use
3) Determine how to extract the performance after changing the learning rate
4) Determine the optimal learning rate
5) Determine how to pull the information about the attention heads
6) Compare the untrained, unoptimized results to our fine-tuned model to draw some conclusion about how BERT learns.