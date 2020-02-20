from nion.typeshed import API_1_0
from libertem.io import dataset
import copy
import typing

class LiberTEMAdapter:
    def __init__(self, api: API_1_0.API, executor):
        self.__api = api
        self.__executor = executor

    def niondata_to_libertemdata(self, niondata: API_1_0.DataItem) -> dataset.base.DataSet:
        sig_dims = niondata.xdata.datum_dimension_count
        data_item = niondata._data_item
        filepath = data_item.persistent_storage.get_storage_property(data_item, 'file_path')
        
        file_parameters = {'path': filepath, 'ds_path': 'data', 'sig_dims': sig_dims}
        
        metadata = copy.deepcopy(niondata.metadata)
        libertem_metadata = {'display_slice': {'start': 0, 'stop': 0}, 'file_parameters': file_parameters}
        metadata['libertem-io'] = copy.deepcopy(libertem_metadata)
        metadata['libertem-io']['file_parameters']['type'] = 'hdf5'
        niondata.set_metadata(metadata)
        
        return dataset.load('hdf5', self.__executor.ensure_sync(), **file_parameters)
