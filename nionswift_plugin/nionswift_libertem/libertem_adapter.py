from nion.typeshed import API_1_0
from libertem.io import dataset

class LiberTEMAdapter:
    def __init__(self, api: API_1_0.API, executor):
        self.__api = api
        self.__executor = executor

    def niondata_to_libertemdata(self, niondata):
        sig_dims = niondata.xdata.datum_dimension_count
        niondata = niondata._data_item
        filepath = niondata.persistent_storage.get_storage_property(niondata, "file_path")

        file_parameters = {"path":filepath, "ds_path":"data", "sig_dims":sig_dims}
        return dataset.load("hdf5", self.__executor.ensure_sync(), **file_parameters)
