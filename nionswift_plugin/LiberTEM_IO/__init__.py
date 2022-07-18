"""
    An importer using LiberTEM as backend

"""

# standard libraries
import gettext
import logging
import threading

import numpy as np


from libertem.io import dataset
from libertem.udf.raw import PickUDF
from libertem.udf.base import UDFRunner
# local libraries
from nion.utils import Registry
from nion.ui import Declarative
from nion.swift.model import PlugInManager
from nion.typeshed import API_1_0, UI_1_0

from .OpenFileDialog import OpenFileDialogUI


_ = gettext.gettext


class LiberTEMIODelegate:

    def __init__(self, api: API_1_0.API, ui: UI_1_0.UserInterface):
        self.__api = api
        self.__ui = ui
        self.io_handler_id = 'libertem_IO_handler'
        self.io_handler_name = 'LiberTEM'
        self.io_handler_extensions = list(dataset.get_extensions())
        self.__file_param_dialog_closed_event = threading.Event()
        self.__file_param_dialog_closed_event.set()
        self.__show_file_param_dialog_finished_event = threading.Event()

    def show_file_param_dialog(self, file_ext: str=None, params_callback: callable=None):
        if self.__file_param_dialog_closed_event.is_set():
            document_controller = self.__api.application.document_controllers[0]._document_controller
            ui_handler = OpenFileDialogUI().get_ui_handler(api_broker=PlugInManager.APIBroker(),event_loop=document_controller.event_loop,file_ext=file_ext,title='File')
            def dialog_closed():
                self.__file_param_dialog_closed_event.set()
            ui_handler.on_closed = dialog_closed

            ui_handler.params_callback = params_callback

            finishes = list()
            dialog = Declarative.construct(document_controller.ui, document_controller, ui_handler.ui_view, ui_handler, finishes)
            for finish in finishes:
                finish()
            ui_handler._event_loop = document_controller.event_loop
            if callable(getattr(ui_handler, 'init_handler', None)):
                ui_handler.init_handler()

            dialog.show()

            ui_handler.request_close = dialog.request_close

            self.__file_param_dialog_closed_event.clear()
        self.__show_file_param_dialog_finished_event.set()

    def can_write_data_and_metadata(self, data_and_metadata, extension):
        return False

    def read_data_and_metadata(self, extension, file_path):
        return self.read_data_and_metadata_from_stream(file_path)

    def read_data_and_metadata_from_stream(self, stream):
        context = Registry.get_component('libertem_context')
        executor = context.executor
        if executor is None:
            logging.error('No libertem executor could be retrieved from the Registry.')
            return
        executor = executor.ensure_sync()
        file_parameters = dataset.detect(stream, executor=executor)
        file_type = file_parameters.get('type', None)
        file_parameters = file_parameters.get('parameters', dict())
        if file_type is None:
            file_type = 'raw'
            file_parameters = {'path': stream}
        file_params = dict()
        def params_callback(file_params_):
            file_params.update(file_params_)

        self.__api.queue_task(lambda: self.show_file_param_dialog(file_type, params_callback))
        self.__show_file_param_dialog_finished_event.wait()
        self.__show_file_param_dialog_finished_event.clear()
        self.__file_param_dialog_closed_event.wait()
        file_params.pop('name', None)
        file_parameters.update(file_params)

        ds = context.load(file_type, **file_parameters)
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
