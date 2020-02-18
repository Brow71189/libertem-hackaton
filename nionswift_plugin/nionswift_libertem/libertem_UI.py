import typing
import gettext
import asyncio
import os
import multiprocessing
import sys

import numpy as np

from nion.utils import Event, Registry
from nion.ui import Declarative
from nion.swift.model import PlugInManager
from nion.swift import Workspace, DocumentController, Panel, Facade
from nion.typeshed import API_1_0

from libertem.executor.base import AsyncAdapter
from libertem.executor.dask import DaskJobExecutor
from libertem.udf.base import UDFRunner, UDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.io.dataset import load


_ = gettext.gettext


def show_display_item(document_window, display_item):
    for display_panel in document_window._document_window.workspace_controller.display_panels:
        if display_panel.display_item == display_item:
            display_panel.request_focus()
            return
    result_display_panel = document_window._document_window.next_result_display_panel()
    if result_display_panel:
        result_display_panel.set_display_panel_display_item(display_item)
        result_display_panel.request_focus()




class LiberTEMUIHandler:
    def __init__(self, api: API_1_0.API, event_loop: asyncio.AbstractEventLoop, ui_view: dict):
        self.ui_view = ui_view
        self.__api = api
        self.__event_loop = event_loop
        self.property_changed_event = Event.Event()
        
    
    def init_handler(self):
        # Needed for method "spawn" (on Windows) to prevent mutliple Swift instances from being started
        if multiprocessing.get_start_method() == 'spawn':
            multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
        self.executor = self.get_libertem_executor()
        Registry.register_component(self.executor, {'libertem_executor'})

    def close(self):
        ...

    

    def get_libertem_executor(self):
        executor = DaskJobExecutor.make_local()
        return AsyncAdapter(wrapped=executor)

    def load_data(self, *args, **kwargs):
        sync_executor = self.executor._wrapped
        ds = sync_executor.run_function(load, *args, **kwargs)
        ds = ds.initialize(sync_executor)
        ds.set_num_cores(len(sync_executor.get_available_workers()))
        sync_executor.run_function(ds.check_valid)
        return ds

    async def run_udf(self, udf: UDF, dataset, roi=None):
        result_iter = UDFRunner(udf).run_for_dataset_async(
            dataset, self.executor, roi=roi, cancel_id="42",
        )

        data_item = None

        async for result in result_iter:
            result_array = np.swapaxes(np.array(result['intensity']), -1, 0)
            if data_item is None:
                data_item = self.show_results(result_array=result_array)
            else:
                xdata = self.get_xdata_for_results(result_array)
                data_item.set_data_and_metadata(xdata)

    def get_xdata_for_results(self, result_array):
        data_descriptor = self.__api.create_data_descriptor(True, 0, 2)
        dimensional_calibrations = [
                self.__api.create_calibration(),
                self.__api.create_calibration(),
                self.__api.create_calibration()
        ]
        intensity_calibration = self.__api.create_calibration()
        xdata = self.__api.create_data_and_metadata(result_array, dimensional_calibrations=dimensional_calibrations,
                                                    intensity_calibration=intensity_calibration,
                                                    data_descriptor=data_descriptor)
        return xdata

    def get_data_item_for_results(self, xdata):
        data_item = self.__api.library.create_data_item_from_data_and_metadata(xdata, title='Result')
        return data_item

    def show_results(self, result_array):
        xdata = self.get_xdata_for_results(result_array)
        data_item = self.get_data_item_for_results(xdata)
        document_controller = self.__api.application.document_controllers[0]._document_controller
        document_window = self.__api.application.document_controllers[0]
        display_item = document_controller.document_model.get_display_item_for_data_item(data_item)
        show_display_item(document_window, display_item)
        return data_item

    def open_button_clicked(self, widget: Declarative.UIWidget):
        file_path = self.file_path_field.text
        if file_path.endswith(('h5', 'hdf5')) and os.path.isfile(file_path):
            ds = self.load_data(
                "hdf5",
                path=file_path,
                ds_path="4DSTEM_experiment/data/datacubes/polyAu_4DSTEM/data",
                min_num_partitions=8,
            )
            udf = ApplyMasksUDF(mask_factories=[lambda: np.ones(ds.shape.sig)])
            self.__event_loop.create_task(self.run_udf(udf, dataset=ds))
            

class LiberTEMUI:
    def __init__(self):
        self.panel_type = 'libertem-panel'

    def get_ui_handler(self, api_broker: PlugInManager.APIBroker=None, event_loop: asyncio.AbstractEventLoop=None, **kwargs):
        api = api_broker.get_api('~1.0')
        ui = api_broker.get_ui('~1.0')
        ui_view = self.__create_ui_view(ui)
        return LiberTEMUIHandler(api, event_loop, ui_view)

    def __create_ui_view(self, ui: Declarative.DeclarativeUI) -> dict:
        open_button = ui.create_push_button(text='Open', on_clicked='open_button_clicked')
        file_path_field = ui.create_line_edit(name='file_path_field')
        open_row = ui.create_row(file_path_field, open_button, ui.create_stretch(), spacing=8, margin=4)
        content = ui.create_column(open_row, ui.create_stretch(), spacing=8, margin=4)

        return content


class LiberTEMPanel(Panel.Panel):
    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: dict):
        super().__init__(document_controller, panel_id, 'libertem-panel')
        panel_type = properties.get('panel_type')
        for component in Registry.get_components_by_type('libertem-panel'):
            if component.panel_type == panel_type:
                ui_handler = component.get_ui_handler(api_broker=PlugInManager.APIBroker(), event_loop=document_controller.event_loop)
                self.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler)

def run():
    Registry.register_component(LiberTEMUI(), {'libertem-panel'})
    panel_properties = {'panel_type': 'libertem-panel'}
    Workspace.WorkspaceManager().register_panel(LiberTEMPanel, 'libertem-processing-panel', _('LiberTEM'), ['left', 'right'], 'right', panel_properties)
