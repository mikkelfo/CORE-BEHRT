import shutil

# Cleanup from setup.py
shutil.rmtree("build")
shutil.rmtree("ehr2vec.egg-info")
# Clean up from test_run.sh
shutil.rmtree("data/test_extra_dir")
shutil.rmtree("data/test_processed")
shutil.rmtree("runs/test_run_pretraining")
shutil.rmtree("runs/test_run_finetuning")
