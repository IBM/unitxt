import inspect
import os
import importlib
import inspect

from .artifact import Artifact
from .utils import Singleton
# Usage
non_registered_files = ['__init__.py', 'artifact.py', 'utils.py', 'register.py', 'metric.py', 'dataset.py', 'blocks.py']

def _register_all_artifacts():
    
    dir = os.path.dirname(__file__)
    file_name = os.path.basename(__file__)
    
    for file in os.listdir(dir):
        if file.endswith('.py') and file not in non_registered_files and file != file_name:
            module_name = file.replace('.py', '')
            
            module = importlib.import_module('.' + module_name, __package__)
            
            for name, obj in inspect.getmembers(module):
                # Make sure the object is a class
                if inspect.isclass(obj):
                    # Make sure the class is a subclass of Artifact (but not Artifact itself)
                    if issubclass(obj, Artifact) and obj is not Artifact:
                        Artifact.register_class(obj)
            

class ProjectArtifactRegisterer(Singleton):
    
    def __init__(self):
        
        if not hasattr(self, '_registered'):
            self._registered = False
        
        if not self._registered:
            _register_all_artifacts()
            self._registered = True
            

def register_all_artifacts():
    ProjectArtifactRegisterer()