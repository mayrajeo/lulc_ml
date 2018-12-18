from keras import Model
from keras.utils import multi_gpu_model


class ModelMGPU(Model):
    """Class for enabling working modelcheckpoints with 
    multigpu models
   """
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
             return getattr(self._smodel, attrname)
    
        return super(ModelMGPU, self).__getattribute__(attrname)
