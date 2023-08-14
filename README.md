To reproduce the results you can run the main scripts in the following order:
-   main_data_pretrain/main_data_pretrain_large: produces raw and tokenized features
-   main_pretrain: pretrain a standard bert model
-   main_data_finetune/main_data_finetune_large: prepare data for finetune, creates outcomes
-   main_finetune: finetune model on a binary task 
To run the hierarchical version:
- setup_hierarchical: uses features produced by main_data_pretrain/main_data_pretrain_large to build a tree and create hierarchical features
- main_h_pretrain: trains a bert model with hierarchical loss
To evaluate using a RF (pretrained model and tokenized features required):
- main_encode_censored_patients: Creates encodings of sequences using a trained model, censored on a certain event
- main_evaluate_rf: Uses CV with an hyperparameter optimized RF to train for a binary clasification task 
  