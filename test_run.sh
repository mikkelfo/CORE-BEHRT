#!/bin/bash

pip install .
python3 -c "import os; os.makedirs('data/test_processed'); os.makedirs('data/test_extra_dir')"

python3 src/scripts/main_data.py loader.data_dir=data/synthea paths.data_dir=data/test_processed paths.extra_dir=data/test_extra_dir
python3 src/scripts/main_pretrain.py trainer_args.epochs=1 trainer_args.effective_batch_size=32 +trainer_args.run_name=test_run_pretraining paths.data_dir=data/test_processed paths.extra_dir=data/test_extra_dir
python3 src/scripts/main_finetune.py paths.pretraining.dir=test_run_pretraining paths.pretraining.model_file=checkpoint_epoch0_end.pt trainer_args.effective_batch_size=32 trainer_args.epochs=1 +trainer_args.run_name=test_run_finetuning paths.data_dir=data/test_processed paths.extra_dir=data/test_extra_dir

python3 cleanup.py