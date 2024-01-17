import sys
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig


ws = Workspace.from_config()
job_name = sys.argv[1]
# Assuming additional arguments are passed after the job name
script_arguments = sys.argv[2:]

env = Environment.get(ws, "PHAIR")

ct = "GPU-A100-small"
#ct = "CPU-MEMORY-OPTIMIZED-RAM112GB-LP"

src = ScriptRunConfig(source_directory=".", script=job_name+".py", compute_target=ct, environment=env,
    arguments=script_arguments)

Experiment(ws, name="experiment_name").submit(src)

