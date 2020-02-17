import typing
import gettext
import asyncio

from nion.utils import Event, Registry
from nion.ui import Declarative
from nion.swift.model import PlugInManager
from nion.swift import Workspace, DocumentController, Panel, Facade
from nion.typeshed import API_1_0


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

    def open_button_clicked(widget: Declarative.UIWidget):
        ...
    
    def init_handler(self):
        ...

    def close(self):
        ...

    

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
        file_path_field = ui.create_line_edit()
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
    Workspace.WorkspaceManager().register_panel(LiberTEMPanel, 'libertem-processing-panel', _('LibertTEM'), ['left', 'right'], 'right', panel_properties)
