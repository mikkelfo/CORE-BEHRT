import importlib
import time
import json

from azureml.core import Model as AzureModel
from . import workspace, Run, log

class Model:
    def __init__(self):
        self._name       = None
        self._version    = None
        self._properties = dict()
        self._tags       = dict()
        self._path       = None
        self._meta_only  = False

    def name(self):
        return self._name
    def version(self):
        return self._version
    def properties(self):
        return self._properties
    def tags(self):
        return self._tags
    def path(self):
        return self._path

    def save(self, path):
        if self._save(path):
            self._path = path
            with open(path+"meta.json", "w") as f:
                f.write(json.dumps({
                    "name":self.name(),
                    "version":self.version(),
                    "properties":self.properties(),
                    "tags":self.tags(),
                    "path":self.path()
                }))
            log().info(f"Model saved to '{path}'!")
        else:
            log().error(f"Could not save model to '{path}'!")
            raise Exception(f"Could not save model to '{path}'!")

    @staticmethod
    def load(model_name, version=None, meta_only=False):
        ws = workspace()
        meta = AzureModel(ws, model_name, version=version)
        log().debug(f"Found model meta: {meta.name}:{meta.version}")
        path = AzureModel.get_model_path(model_name, version=version, _workspace=ws)
        
        properties = meta.properties

        if "AutoML-NLP" in meta.tags:
            log().debug(f"AutoML-NLP tag found - treating as AutoML model...")
            properties = {
                "class":"AzureAutoMLClassifier",
                "module":"aiomic.ml.nlp",
                "index":"Note_ID",
                "columns":"Note",
                "target":meta.tags["target"]
            }
            
        if "class" not in properties or "module" not in properties:
            raise Exception("Unknown model type - cannot load...")
        
        cls = properties["class"]
        mod = properties["module"]
        log().debug(f"Loading class {cls} from module {mod}...")
        ModelCls = getattr(importlib.import_module(mod), cls)
        
        # Create model and populate meta
        model             = ModelCls(**properties)
        model._name       = meta.name
        model._version    = meta.version
        model._tags       = meta.tags
        model._properties = properties
        model._path       = path

        if meta_only:
            model._meta_only = True
        else:
            model._load(path)
        
        return model

    def register(self, name=None, exist_ok=False, path=None, datasets=[]):
        ws = workspace()
    
        if name is None and self.name() is None:
            raise Exception("Cannot register model - no name provided and model is not previously named!")

        if name is not None:
            self._name = name
        
        # Check exists if required
        if not exist_ok:
            try:
                meta = Model.load(self.name(), meta_only=True)
                raise Exception(f"Model {self.name()} already registered, version = {meta.version()}!")
            except:
                pass # All ok, model not found!

        # Save if needed
        if path is not None:
            self.save(path)
        
        # Check that model is saved!
        if self.path() is None:
            raise Exception(f"Cannot register an unsaved model! Please call save beforehand or provide a path...")

        self._properties["class"]  = self.__class__.__name__
        self._properties["module"] = self.__class__.__module__
        
        log().info(f"Registering model {self.name()}, path='{self.path()}'")
    
        if Run.is_remote():
            # Wait 20 secs here - to ensure files are uploaded to run.
            log().debug("Sleeping...")
            time.sleep(30)
            log().debug("Done sleeping!")
            # Register in run
            meta = Run.register_model(
                model_name=self.name(),
                model_path=self.path(),
                datasets=datasets,
                tags=self.tags(),
                properties=self.properties(),
            )
        else:
            # Just register in workspace
            meta = AzureModel.register(
                workspace=ws,
                model_name=self.name(),
                model_path=self.path(),
                datasets=datasets,
                tags=self.tags(),
                properties=self.properties(),
            )
        
        self._version = meta.version

    # Subclasses must override
    def _load(self, path):
        raise Exception("Incomplete model implementation - _load not implemented!")
    
    def _save(self, path):
        raise Exception("Incomplete model implementation - _save not implemented!")


