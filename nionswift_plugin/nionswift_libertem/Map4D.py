# system imports
import gettext
import logging
import numpy as np
import time
import copy
import functools
import json
import threading

# local libraries
from nion.typeshed import API_1_0 as API
from nion.data import xdata_1_0 as xd
from nion.swift import Facade
from nion.swift.model import Symbolic
from nion.utils import Binding, Registry

from libertem.io import dataset
from libertem.udf.base import UDFRunner, UDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.raw import PickUDF
from libertem.executor.base import JobCancelledError

from .libertem_adapter import LiberTEMAdapter

_ = gettext.gettext


class Map4D:
    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__api = computation.api
        self.__event_loop = self.__api.application.document_controllers[0]._document_controller.event_loop
        
        def create_panel_widget(ui, document_controller):
            def select_button_clicked():
                graphics = Facade.DataItem(self.computation._computation.source).graphics
                if not graphics:
                    return
                graphics_variable = self.computation._computation._get_variable('map_regions')
                graphics_variable.disconnect_items()
                if graphics_variable.bound_items_model is None:
                    return
                num_items = len(graphics_variable.bound_items_model.items)
                for _ in range(num_items):
                    self.computation._computation.remove_item_from_objects('map_regions', 0)
                for graphic in graphics:
                    if graphic._graphic.role == 'mask':
                        self.computation._computation.insert_item_into_objects('map_regions', 0, Symbolic.make_item(graphic._graphic, type='graphic'))

            column = ui.create_column_widget()
            row = ui.create_row_widget()

            select_graphics_button = ui.create_push_button_widget('Update masks')
            row.add_spacing(10)
            row.add(select_graphics_button)
            row.add_stretch()
            row.add_spacing(10)

            column.add_spacing(10)
            column.add(row)
            column.add_spacing(10)
            column.add_stretch()

            select_graphics_button.on_clicked = select_button_clicked

            return column
        # Disable mask updating for now because it is broken
        # self.computation._computation.create_panel_widget = create_panel_widget
        
    def get_xdata_for_results(self, result_array):
        data_descriptor = self.__api.create_data_descriptor(False, 0, 2)
        dimensional_calibrations = [
                self.__api.create_calibration(),
                self.__api.create_calibration()
        ]
        intensity_calibration = self.__api.create_calibration()
        xdata = self.__api.create_data_and_metadata(result_array, dimensional_calibrations=dimensional_calibrations,
                                                    intensity_calibration=intensity_calibration,
                                                    data_descriptor=data_descriptor)
        return xdata
        
    async def run_udf(self, udf: UDF, cancel_id, executor, dataset, roi=None):
        try:
            result_iter = UDFRunner(udf).run_for_dataset_async(
                dataset, executor, roi=roi, cancel_id=cancel_id,
            )
    
            async for result in result_iter:
                result_array = np.squeeze(np.swapaxes(np.array(result['intensity']), -1, 0))
                self.__new_xdata = self.get_xdata_for_results(result_array)
                self.commit()
        except JobCancelledError:
            pass
            
    def execute(self, src, map_regions):
        try:
            if hasattr(self.computation._computation, 'last_src_uuid') and hasattr(self.computation._computation, 'last_map_regions'):
                map_regions_ = [region.persistent_dict for region in map_regions]
                if str(src.uuid) == self.computation._computation.last_src_uuid and map_regions_ == self.computation._computation.last_map_regions:
                    return
                if str(src.uuid) != self.computation._computation.last_src_uuid and hasattr(self.computation._computation, 'ds'):
                    self.computation._computation.ds = None
            metadata = copy.deepcopy(src.xdata.metadata)
            libertem_metadata = metadata.get('libertem-io')
            if libertem_metadata is None:
                return
            executor = Registry.get_component('libertem_executor')
            if executor is None:
                logging.error('No libertem executor could be retrieved from the Registry.')
                return
            file_parameters = libertem_metadata['file_parameters']
            file_type = file_parameters.pop('type')
            shape = src.xdata.datum_dimension_shape
            if map_regions:
                mask_data = np.zeros(shape, dtype=np.bool)
                for region in map_regions:
                    np.logical_or(mask_data, region.get_mask(shape), out=mask_data)
            else:
                mask_data = np.ones(shape, dtype=np.bool)
            if hasattr(self.computation._computation, 'ds') and self.computation._computation.ds:
                ds = self.computation._computation.ds
            else:
                ds = dataset.load(file_type, executor.ensure_sync(), **file_parameters)
                self.computation._computation.ds = ds
            udf = ApplyMasksUDF(mask_factories=[lambda: mask_data])
            dc = self.__api.application.document_controllers[0]._document_controller
            if hasattr(self.computation._computation, 'cancel_id'):
                to_cancel = self.computation._computation.cancel_id
                self.__api.queue_task(lambda: self.__event_loop.create_task(executor.cancel(to_cancel)))
                #self.computation._computation.cancel_id = None
            self.computation._computation.cancel_id = str(time.time())
            dc.add_task('libertem-map4d', lambda: self.__event_loop.create_task(self.run_udf(udf, self.computation._computation.cancel_id, executor, dataset=ds)))
            self.computation._computation.last_src_uuid = str(src.uuid)
            self.computation._computation.last_map_regions = copy.deepcopy([region.persistent_dict for region in map_regions])
            
        except Exception:
            import traceback
            traceback.print_exc()

    def commit(self):
        try:
            self.computation.set_referenced_xdata('target', self.__new_xdata)
        except AttributeError:
            pass


class Map4DMenuItem:

    menu_id = 'libertem_menu'  # required, specify menu_id where this item will go
    menu_name = _('LiberTEM') # optional, specify default name if not a standard menu
    menu_before_id = 'window_menu' # optional, specify before menu_id if not a standard menu
    menu_item_name = _('Map 4D')  # menu item name

    def __init__(self, api):
        self.__api = api
        def init():
            document_controller = self.__api.application.document_controllers[0]._document_controller
            computation_data_items = document_controller.ui.get_persistent_string('libertem_map4d_data_items_0')
            self.__computation_data_items = json.loads(computation_data_items) if computation_data_items else dict()
            self.__tool_tip_boxes = list()
            document_model = self.__api.application._application.document_model
            for computation in document_model.computations:
                src = computation.get_input('src')
                if src and self.__computation_data_items.get(str(src.uuid)) == 'source':
                    target = computation.get_output('target')
                    if target is None:
                        continue
                    target_api = Facade.DataItem(target)
                    pick_graphic = None
                    for graphic in target_api.graphics:
                        if graphic.label == 'Pick':
                            pick_graphic = graphic
                            break
                    if pick_graphic is not None:
                        self.__connect_pick_graphic(Facade.DataItem(src), target_api, pick_graphic, computation)
            self.__display_item_changed_event_listener = (
                               document_controller.focused_display_item_changed_event.listen(self.__display_item_changed))
            
        def schedule_init():
            while not self.__api.application.document_controllers:
                time.sleep(0.5)
            self.__api.queue_task(init)
            
        threading.Thread(target=schedule_init, daemon=True).start()

    def __display_item_changed(self, display_item):
        data_item = display_item.data_item if display_item else None
        if data_item:
            tip_id = self.__computation_data_items.get(str(data_item.uuid))
            if tip_id:
                self.__show_tool_tips(tip_id)

    def __show_tool_tips(self, tip_id='source', timeout=30):
        for box in self.__tool_tip_boxes:
            box.remove_now()
        self.__tool_tip_boxes = list()
        if tip_id == 'source':
            text = ('Click "Select" in the computation panel (Window -> Computation) to update the masks used for this '
                    'computation. All graphics tagged as mask (Processing -> Graphics -> Add to Mask) will be used.'
                    '\nWithout a mask, the whole frames will be summed.')
        elif tip_id == 'map_4d':
            text = 'Move the "Pick" graphic to change the data slice in the source data item.'
        elif tip_id == 'wrong_shape':
            text = 'This computation only works for 4D-data.'
        else:
            return
        document_controller = self.__api.application.document_windows[0]
        workspace = document_controller._document_controller.workspace_controller
        box = workspace.pose_tool_tip_box(text, timeout)
        #box = document_controller.show_tool_tip_box(text, timeout)
        self.__tool_tip_boxes.append(box)
        
    def __connect_pick_graphic(self, src, target, pick_graphic, computation, do_wait=-1):
        def _update_collection_index(axis, value):
            if src.xdata.is_collection or src.xdata.is_sequence:
                display_item = self.__api.application._application.document_model.get_display_item_for_data_item(src._data_item)
                collection_index = display_item.display_data_channel.collection_index
                if axis == 0:
                    if value != collection_index[0]:
                        display_item.display_data_channel.collection_index = (value, collection_index[1], 0)
                else:
                    if value != collection_index[1]:
                        display_item.display_data_channel.collection_index = (collection_index[0], value, 0)
            else:
                libertem_metadata = copy.deepcopy(src.metadata.get('libertem-io'))
                if not libertem_metadata:
                    return
                file_parameters = libertem_metadata['file_parameters']
                file_type = file_parameters.pop('type')
                current_index = libertem_metadata['display_slice']['start']
                current_index = np.unravel_index(current_index, target.data.shape)
                if value == current_index[axis]:
                    return
                executor = Registry.get_component('libertem_executor')
                if not executor:
                    return
                executor = executor.ensure_sync()
                ds = dataset.load(file_type, executor, **file_parameters)
                roi = np.zeros(ds.shape.nav, dtype=bool)
                if axis == 0:
                    roi[value, current_index[1]] = True
                    current_index = (value, current_index[1])
                else:
                    roi[current_index[0], value] = True
                    current_index = (current_index[0], value)
                result = UDFRunner(PickUDF()).run_for_dataset(ds, executor, roi=roi)
                result_array = np.squeeze(np.array(result['intensity']))
                new_metadata = copy.deepcopy(src.metadata)
                new_display_slice = np.ravel_multi_index(current_index, target.data.shape)
                new_metadata['libertem-io']['display_slice']['start'] = new_display_slice
                new_xdata = self.__api.create_data_and_metadata(result_array, metadata=new_metadata)
                src.set_data_and_metadata(new_xdata)
        
        if do_wait > 0:
            starttime = time.time()
            while target.data is None:
                if time.time() - starttime > do_wait:
                    break
                time.sleep(0.1)
        
        if target.data is None:
            return
        shape = target.data.shape
        computation.pick_graphic_binding_0 = Binding.TuplePropertyBinding(pick_graphic._graphic, 'position', 0, converter=FloatTupleToIntTupleConverter(shape[0], 0))
        computation.pick_graphic_binding_1 = Binding.TuplePropertyBinding(pick_graphic._graphic, 'position', 1, converter=FloatTupleToIntTupleConverter(shape[1], 1))
        computation.pick_graphic_binding_0.target_setter = functools.partial(_update_collection_index, 0)
        computation.pick_graphic_binding_1.target_setter = functools.partial(_update_collection_index, 1)
        
        def collection_index_changed(key):
            if src.xdata.is_collection:
                display_item = self.__api.application._application.document_model.get_display_item_for_data_item(src._data_item)
                if key == 'collection_index':
                    collection_index = display_item.display_data_channel.collection_index
                    if int(pick_graphic.position[0]*shape[0]) != collection_index[0]:
                        computation.pick_graphic_binding_0.update_source(collection_index)
                    if int(pick_graphic.position[1]*shape[1]) != collection_index[1]:
                        computation.pick_graphic_binding_1.update_source(collection_index)
        if src.xdata.is_collection:
            display_item = self.__api.application._application.document_model.get_display_item_for_data_item(src._data_item)
            computation.collection_index_changed_event_listener = display_item.display_data_channel.property_changed_event.listen(collection_index_changed)

        

    def menu_item_execute(self, window: API.DocumentWindow) -> None:
        document_controller = window._document_controller
        selected_display_item = document_controller.selected_display_item
        data_item = (selected_display_item.data_items[0] if
                     selected_display_item and len(selected_display_item.data_items) > 0 else None)

        if data_item:
            api_data_item = Facade.DataItem(data_item)
            ds = None
            if not api_data_item.xdata.metadata.get('libertem-io'):
                executor = Registry.get_component('libertem_executor')
                if not executor:
                    return
                ds = LiberTEMAdapter(self.__api, executor).niondata_to_libertemdata(api_data_item)
                if not api_data_item.xdata.metadata.get('libertem-io'):
                    self.__show_tool_tips('wrong_shape')
                    return
            map_data_item = self.__api.library.create_data_item(title='Map 4D of ' + data_item.title)
            display_item = document_controller.document_model.get_display_item_for_data_item(map_data_item._data_item)
            show_display_item(window, display_item)
            map_regions = list()
            for graphic in api_data_item.graphics:
                if graphic._graphic.role == 'mask':
                    map_regions.append(graphic)
            computation = self.__api.library.create_computation('nion.libertem.map_4d',
                                                                inputs={'src': api_data_item,
                                                                        'map_regions': map_regions},
                                                                outputs={'target': map_data_item})
            computation._computation.source = data_item
            if ds is not None:
                computation._computation.ds = ds

            map_display_item = document_controller.document_model.get_display_item_for_data_item(map_data_item)
            document_controller.show_display_item(map_display_item)
            pick_graphic = map_data_item.add_point_region(0.5, 0.5)
            pick_graphic.label = 'Pick'
            
            
            threading.Thread(target=self.__connect_pick_graphic, args=(api_data_item, map_data_item, pick_graphic, computation._computation, 30), daemon=True).start()
            
            self.__computation_data_items.update({str(data_item.uuid): 'source',
                                                  str(map_data_item._data_item.uuid): 'map_4d'})
            self.__api.application.document_controllers[0]._document_controller.ui.set_persistent_string('libertem_map4d_data_items_0', json.dumps(self.__computation_data_items))
            self.__show_tool_tips()


def show_display_item(document_window, display_item):
    for display_panel in document_window._document_window.workspace_controller.display_panels:
        if display_panel.display_item == display_item:
            display_panel.request_focus()
            return
    result_display_panel = document_window._document_window.next_result_display_panel()
    if result_display_panel:
        result_display_panel.set_display_panel_display_item(display_item)
        result_display_panel.request_focus()
        
        
class FloatTupleToIntTupleConverter:
    def __init__(self, axis_size, axis_index):
        self.axis_size = axis_size
        self.axis_index = axis_index

    def convert(self, value):
        return int(value*self.axis_size)

    def convert_back(self, value):
        return (value[self.axis_index] + 0.5)/self.axis_size


class Map4DExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.extension.libertem_map_4d"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        # be sure to keep a reference or it will be closed immediately.
        self.__menu_item_ref = api.create_menu_item(Map4DMenuItem(api))

    def close(self):
        self.__menu_item_ref.close()

Symbolic.register_computation_type('nion.libertem.map_4d', Map4D)
