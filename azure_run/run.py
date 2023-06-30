import json
import time
import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix
from azureml.core import Run as AzureRun, ScriptRunConfig, Environment

from . import workspace, log

class Run:
    _INSTANCE = None

    MAX_ACTIVE_CHILDREN = 5

    def __init__(self, remote=None, callback=None):
        self._children = dict()

        self.remote    = remote
        self.callback  = callback
        self._seed     = None

    @staticmethod
    def init():
        if Run._INSTANCE is None:
            # Check if we are in Azure context
            remote = None
            try:
                remote = AzureRun.get_context(allow_offline=False)
            except:
                pass
            Run._INSTANCE = Run(remote=remote)
        return Run._INSTANCE

    @staticmethod
    def is_remote():
        R = Run.init()
        return R.remote is not None

    @staticmethod
    def name(name=None):
        R = Run.init()
        if R.remote is None:
            return None
        else:
            if name is not None:
                R.remote.display_name = name
                log().info(f"Run name set = {name}!")
            return R.remote.display_name

    @staticmethod
    def seed(seed=None):
        R = Run.init()
        if seed is not None:
            if R._seed is not None:
                log().warning(f"Seed already set (= {R._seed}), ignoring new seed {seed}...")
            else:
                R._seed = seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                Run.log_metric("Seed", seed)
                log().info(f"Seed set = {R._seed}!")
        return R._seed

    @staticmethod
    def submit_child(
            script,
            arguments=[],
            callback=None,
            name=None,
            tags=None):
        R = Run.init()
        if R.remote is None:
            raise Exception("Local childs not supported yet...")
        
        # Wait until there are less than MAX_ACTIVE_CHILDREN.
        while Run.active_children()>=Run.MAX_ACTIVE_CHILDREN:
            time.sleep(5)

        # There are +1 available spots, create run
        ws = workspace()
        env = R.remote.get_environment()
        ct  = "local"
        src = ScriptRunConfig(source_directory=".", script=script, arguments=arguments, compute_target=ct, environment=env)
        aRc = R.remote.submit_child(src, tags=tags)
        if name is not None: aRc.display_name = name
        log().debug(f"Child run started, name = {name}.")
        Rc = Run(remote=aRc, callback=callback)
        rid = aRc.get_details()["runId"]

        R._children[rid] = Rc
    
    @staticmethod
    def active_children():
        R = Run.init()
        Run.join_children(block=False)
        return len(R._children)

    @staticmethod
    def join_children(block=True):
        R = Run.init()
        joined = 0
        rnd = 0
        while len(R._children)>0:
            rnd += 1
            ncmp = dict()
            for rid,Rc in R._children.items():
                status = Rc.remote.get_status()
                if status in ("Completed","Failed","Canceled"):
                    metrics = Rc.remote.get_metrics()
                    tags    = Rc.remote.get_tags()
                    # Callback
                    log().debug(f"Child joined! RID = {rid}, tags = {tags}")
                    if Rc.callback is not None:
                        Rc.callback(rid, status, metrics=metrics, tags=tags)
                    joined += 1
                else:
                    ncmp[rid] = Rc
            R._children = ncmp
            if not block: break
            log().debug(f"Waiting for children to join ({len(R._children)}), sleeping...")
            time.sleep(5)
        
        return joined

    @staticmethod
    def register_model(model_name, model_path, datasets=[], tags=dict(), properties=dict()):
        R = Run.init()
        if R.remote is not None:
            return R.remote.register_model(
                model_name=model_name,
                model_path=model_path,
                datasets=datasets,
                tags=tags,
                properties=properties
            )
        else:
            raise Exception(f"Error: cannot register model with run - no remote run...")

    @staticmethod
    def log_metric(name, value):
        R = Run.init()
        if R.remote is not None:
            R.remote.log(name, value)
        else:
            log().info(f"Metric logged: {name} = {value}")

    @staticmethod
    def log_row(name, description=None, **kwargs):
        R = Run.init()
        if R.remote is not None:
            R.remote.log_row(name, description=description, **kwargs)
        else:
            log().info(f"Row logged: {name} = {kwargs}")

    @staticmethod
    def log_confusion_matrix(Y, Pr, threshold=0.5, name="Confusion matrix"):
        R = Run.init()
        P = Pr.copy()
        P[P>=threshold] = 1
        P[P<threshold]  = 0
        tn, fp, fn, tp = confusion_matrix(Y, P.astype(int), labels=[0,1]).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        matrix = [
            [tn, fp],
            [fn, tp]
        ]
        if R.remote is not None:
            value = {
                "schema_type": "confusion_matrix",
                "schema_version": "1.0.0",
                "data": {
                    "class_labels": ["0","1"],
                    "matrix":matrix,
                }
            }
            R.remote.log_confusion_matrix(name, json.dumps(value))
        else:
            log().info(f"Confusion matrix logged: {matrix}")

    @staticmethod
    def log_evaluation(Y, Pr, name="Evaluation", num_thresholds=40):
        R = Run.init()
        if R.remote is not None:
            bsize = 1/(num_thresholds-1)
            pr_ts = list(np.arange(0,1+bsize/2,bsize))
            pe_ts = [np.quantile(Pr, t) for t in pr_ts]
            def _cm(t, flip=False):
                P = Pr.copy()
                P[P>=t] = 1
                P[P<t]  = 0
                tn, fp, fn, tp = confusion_matrix(Y, P.astype(int), labels=[0,1]).ravel()
                tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
                return [tn, fn, tp, fp] if flip else [tp, fp, tn, fn]

            value = {
                "schema_type": "accuracy_table",
                "schema_version": "1.0.1",
                "data": {
                    "probability_tables": [
                        [_cm(t, False) for t in pr_ts]
                    ],
                    "percentile_tables": [
                        [_cm(t, False) for t in pe_ts]
                    ],
                    "probability_thresholds": pr_ts,
                    "percentile_thresholds": pe_ts,
                    "class_labels": ["1"]
                }
            }
            R.remote.log_accuracy_table(name, json.dumps(value))
        else:
            log().info(f"Attempted to log accuracy table.")

    @staticmethod
    def log_plot(plt, name, filename=None):
        R = Run.init()
        if R.remote is not None:
            R.remote.log_image(name, plot=plt)
            plt.close()
        elif filename is not None:
            plt.savefig(filename)
            plt.close()
            log().info(f"Saved plot to {filename}.")
        else:
            log().warning(f"Attempted to log plot to non-run - skipping. Provide 'filename' to save as file instead.")

    @staticmethod
    def set_tags(tags):
        R = Run.init()
        if R.remote is not None:
            R.remote.set_tags(tags)
        else:
            log().info(f"Tags = {tags}")
    
