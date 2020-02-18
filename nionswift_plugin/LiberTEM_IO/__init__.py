"""
    An importer using LiberTEM as backend

"""

# standard libraries
import gettext
import logging
import warnings

import datetime
import json
import numpy as np


from libertem.io import dataset
from libertem.udf.raw import PickUDF
from libertem.udf.base import UDFRunner
# local libraries
from nion.utils import Registry
from nion.typeshed import API_1_0, UI_1_0


_ = gettext.gettext


class LiberTEMIODelegate:

    def __init__(self, api: API_1_0.API, ui: UI_1_0.UserInterface):
        self.__api = api
        self.__ui = ui
        self.io_handler_id = 'libertem_IO_handler'
        self.io_handler_name = 'LiberTEM'
        self.io_handler_extensions = list(dataset.filetypes.keys()) + ['hdr']
        
    def can_write_data_and_metadata(self, data_and_metadata, extension):
        return False

    def read_data_and_metadata(self, extension, file_path):
        return self.read_data_and_metadata_from_stream(file_path)

    def read_data_and_metadata_from_stream(self, stream):
        executor = Registry.get_component('libertem_executor')
        if executor is None:
            logging.error('No libertem executor could be retrieved from the Registry.')
            return
        executor = executor.ensure_sync()
        file_parameters = dataset.detect(stream, executor=executor)
        file_type = file_parameters.pop('type', None)
        if file_type is None:
            logging.error(f'Cannot load file {stream} with the LiberTEM backend.')
            return
            
        def load_data(*args, **kwargs):
            ds = executor.run_function(dataset.load, *args, **kwargs)
            ds = ds.initialize(executor)
            ds.set_num_cores(len(executor.get_available_workers()))
            executor.run_function(ds.check_valid)
            return ds
        
        ds = load_data(file_type, **file_parameters)
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi_flat = roi.ravel()
        roi_flat[0] = True
        result = UDFRunner(PickUDF()).run_for_dataset(ds, executor, roi=roi)
        result_array = np.squeeze(np.array(result['intensity']))
        file_parameters['type'] = file_type
        metadata = {'libertem-io': {'file_parameters': file_parameters, 'display_slice': {'start': 0, 'stop': 0}}}
        return self.__api.create_data_and_metadata(result_array, metadata=metadata)

    def write_data_item(self, data_item, file_path, extension) -> None:
        self.write_data_item_stream(data_item, file_path)
        
    def write_data_item_stream(self, data_item, stream) -> None:
        self.write_data_and_metadata_stream(data_item.xdata, stream)

    def write_data_and_metadata_stream(self, data_and_metadata, stream) -> None:
        ...


class LiberTEMIOExtension:

    # required for Nion Swift to recognize this as an extension class.
    extension_id = 'nion.swift.extensions.libertem_io'

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version='~1.0')
        ui = api_broker.get_ui(version='~1.0')
        # be sure to keep a reference or it will be closed immediately.
        self.__io_handler_ref = api.create_data_and_metadata_io_handler(LiberTEMIODelegate(api, ui))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__io_handler_ref.close()
        self.__io_handler_ref = None
