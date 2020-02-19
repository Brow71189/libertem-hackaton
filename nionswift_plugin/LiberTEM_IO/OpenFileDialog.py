import typing
import asyncio

from nion.swift.model import PlugInManager
from nion.ui import Declarative
from nion.utils import Event, Converter
from nion.typeshed import API_1_0


class OpenFileDialogUIHandler:
    def __init__(self, api: API_1_0.API, ui_view: dict, params: dict):
        self.__api = api
        self.ui_view = ui_view
        self.property_changed_event = Event.Event()
        self.on_closed = None
        self.params_callback = None
        self.tuple_to_string_converter = TupleToStringConverter()
        self.__file_params = {}

        for i,param in enumerate(params):
            self.__create_params_value(f'params_{i}', param['id'])

    def init_handler(self):
        pass

    def on_load(self, widget: Declarative.UIWidget):
        if callable(self.params_callback):
            self.params_callback(self.__file_params)
        if hasattr(self, 'request_close') and callable(self.request_close):
            self.request_close()

    def close(self):
        if callable(self.on_closed):
            self.on_closed()

    def __create_params_value(self, name: str, property_name: str):
        def getter(self):
            return self.__file_params.get(property_name)

        def setter(self, value):
            print(type(value))
            self.__file_params[property_name] = value
            self.property_changed_event.fire(name)

        setattr(OpenFileDialogUIHandler, name, property(getter, setter))


class OpenFileDialogUI:
    def get_ui_handler(self, api_broker: PlugInManager.APIBroker=None, event_loop: asyncio.AbstractEventLoop=None, file_ext: str=None, **kwargs):
        api = api_broker.get_api('~1.0')
        ui = api_broker.get_ui('~1.0')

        params = {
            'hdf5':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}, {'text': 'HDF5 Dataset Path', 'type': 'text_box', 'id': 'ds_path'}, {'text': 'Tileshape', 'type': 'text_box', 'id': 'tileshape', 'converter': 'tuple_to_string_converter'}],
            'raw':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}, {'text': 'Scan Size', 'type': 'text_box', 'id': 'scan_size', 'converter': 'tuple_to_string_converter'}, {'text': 'Datatype', 'type': 'text_box', 'id': 'dtype'}, {'text': 'Detector Size', 'type': 'text_box', 'id': 'detector_size', 'converter': 'tuple_to_string_converter'}, {'text': 'Enable Direct I/O', 'type': 'check_box', 'id': 'enable_direct'}],
            'mib':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}, {'text': 'Tileshape', 'type': 'text_box', 'id': 'tileshape', 'converter': 'tuple_to_string_converter'}, {'text': 'Scan Size', 'type': 'text_box', 'id': 'scan_size', 'converter': 'tuple_to_string_converter'}],
            'blo':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}, {'text': 'Tileshape', 'type': 'text_box', 'id': 'tileshape', 'converter': 'tuple_to_string_converter'}],
            'k2is':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}],
            'ser':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}],
            'frms6':[{'text': 'Name', 'type': 'text_box', 'id': 'name'}],
            'empad': [{'text': 'Name', 'type': 'text_box', 'id': 'name'}, {'text': 'Scan Size', 'type': 'text_box', 'id': 'scan_size', 'converter': 'tuple_to_string_converter'}]
        }
        file_params = params.get(file_ext)
        ui_view = self.__create_ui_view(ui, title=kwargs.get('title'), file_ext=file_ext, params=file_params)
        return OpenFileDialogUIHandler(api, ui_view, file_params)

    def __create_ui_view(self, ui: Declarative.DeclarativeUI, title: str=None, file_ext: str='hdf5', params:dict=None, **kwargs) -> dict:
        ui_objects = []
        for i,param in enumerate(params):
            converter = ''
            if param.get('converter'):
                converter = ', converter=' + param['converter']
            if(param['type'] == 'text_box'):
                ui_objects.append(ui.create_label(text=param['text']))
                ui_objects.append(ui.create_line_edit(text=f'@binding(params_{i}{converter})'))
            elif(param['type'] == 'check_box'):
                ui_objects.append(ui.create_check_box(text=param['text'], checked=f'@binding(params_{i}{converter})'))

        ui_objects.append(ui.create_row(ui.create_push_button(text='Load', on_clicked='on_load'), ui.create_stretch()))
        content = ui.create_column(*ui_objects, ui.create_stretch(), spacing = 8, margin=4)

        return ui.create_modeless_dialog(content, title=title+' - '+file_ext.title(), margin=4)


class TupleToStringConverter:

    def convert(self, value):
        return str(value)

    def convert_back(self, formatted_value):
        return eval(formatted_value)
