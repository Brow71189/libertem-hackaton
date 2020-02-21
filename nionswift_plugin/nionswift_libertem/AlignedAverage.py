# system imports
import gettext
import logging
import numpy as np
import time
import copy
import json
import threading
from scipy.ndimage import fourier_uniform, shift

# local libraries
from nion.typeshed import API_1_0 as API
from nion.swift import Facade
from nion.swift.model import Symbolic
from nion.utils import Registry, Binding, Converter

from libertem.io import dataset
from libertem.udf.base import UDFRunner, UDF
from libertem.udf.raw import PickUDF
from libertem.executor.base import JobCancelledError

from .libertem_adapter import LiberTEMAdapter

_ = gettext.gettext


def normalized_corr(image, template):
    """
    Correctly normalized template matching by cross-correlation. The result should be the same as what you get from
    openCV's "match_template" function with method set to "ccoeff_normed", except for the output shape, which will
    be image.shape here (as opposed to openCV, where only the valid portion of the image is returned).
    Used ideas from here:
    http://scribblethink.org/Work/nvisionInterface/nip.pdf (which is an extended version of this paper:
    J. P. Lewis, "FastTemplateMatching", Vision Interface, p. 120-123, 1995)
    """
    template = template.astype(np.float64)
    image = image.astype(np.float64)
    normalized_template = template - np.mean(template)
    # inverting the axis of a real image is the same as taking the conjugate of the fourier transform
    fft_normalized_template_conj = np.fft.fft2(normalized_template[::-1, ::-1], s=image.shape)
    fft_image = np.fft.fft2(image)
    fft_image_squared = np.fft.fft2(image**2)
    fft_image_squared_means = fourier_uniform(fft_image_squared, template.shape)
    image_means_squared = (np.fft.ifft2(fourier_uniform(fft_image, template.shape)).real)**2
    # only normalizing the template is equivalent to normalizing both (see paper in docstring for details)
    fft_corr = fft_image * fft_normalized_template_conj
    # we need to shift the result back by half the template size
    shift = (int(-1*(template.shape[0]-1)/2), int(-1*(template.shape[1]-1)/2))
    corr = np.roll(np.fft.ifft2(fft_corr).real, shift=shift, axis=(0,1))
    # use Var(X) = E(X^2) - E(X)^2 to calculate variance
    image_variance = np.fft.ifft2(fft_image_squared_means).real - image_means_squared
    denom = image_variance * template.size * np.sum(normalized_template**2)
    denom[denom<0] = np.amax(denom)
    return corr/np.sqrt(denom)


def parabola_through_three_points(p1, p2, p3):
    """
    Calculates the parabola a*(x-b)**2+c through three points. The points should be given as (y, x) tuples.
    Returns a tuple (a, b, c)
    """
    # formula taken from http://stackoverflow.com/questions/4039039/fastest-way-to-fit-a-parabola-to-set-of-points
    # Avoid division by zero in calculation of s
    if p2[0] == p3[0]:
        temp = p2
        p2 = p1
        p1 = temp

    s = (p1[0]-p2[0])/(p2[0]-p3[0])
    b = (-p1[1]**2 + p2[1]**2 + s*(p2[1]**2 - p3[1]**2)) / (2*(-p1[1] + p2[1] + s*p2[1] - s*p3[1]))
    a = (p1[0] - p2[0]) / ((p1[1] - b)**2 - (p2[1] - b)**2)
    c = p1[0] - a*(p1[1] - b)**2
    return (a, b, c)


def find_ccorr_max(ccorr):
    ccorr = np.squeeze(ccorr)
    max_pos = np.unravel_index(np.argmax(ccorr), ccorr.shape)
    if (np.array(max_pos) < np.ones_like(ccorr.shape)).any() or (np.array(max_pos) > np.array(ccorr.shape) - 2).any():
        return 1, ccorr[max_pos], max_pos
    if ccorr.ndim > 1:
        max_y = ccorr[max_pos[0]-1:max_pos[0]+2, max_pos[1]]
        parabola_y = parabola_through_three_points((max_y[0], max_pos[0]-1),
                                                   (max_y[1], max_pos[0]  ),
                                                   (max_y[2], max_pos[0]+1))
        max_x = ccorr[max_pos[0], max_pos[1]-1:max_pos[1]+2]
        parabola_x = parabola_through_three_points((max_x[0], max_pos[1]-1),
                                                   (max_x[1], max_pos[1]  ),
                                                   (max_x[2], max_pos[1]+1))
        return 0, ccorr[max_pos], (parabola_y[1], parabola_x[1])
    else:
        max_pos = max_pos[0]
        max_x = ccorr[max_pos-1:max_pos+2]
        parabola_x = parabola_through_three_points((max_x[0], max_pos-1),
                                                   (max_x[1], max_pos),
                                                   (max_x[2], max_pos+1))
        return 0, ccorr[max_pos], (parabola_x[1],)
    

class AlignedSumUDF(UDF):
    def __init__(self, reference_frame, crop_slice_tuple=None, ccorr_threshold=0.5, **kwargs):
        super().__init__(reference_frame=reference_frame, crop_slice_tuple=crop_slice_tuple, ccorr_threshold=ccorr_threshold, **kwargs)
    
    def get_result_buffers(self):
        return {'sum_buffer': self.buffer(kind='sig', dtype=np.float32),
                'shifts_buffer': self.buffer(kind='nav', dtype=np.float32, extra_shape=(len(self.meta.dataset_shape.sig.to_tuple()), )),
                'number_processed_frames': self.buffer(kind='single', dtype=int),
                'processed_frame_indices': self.buffer(kind='single', dtype=int, extra_shape=(np.prod(self.meta.dataset_shape.nav), ))}
        
    def calculate_shift(self, image1, image2):
        """
        Calculate shift of image2 with respect to image1.
        Image2 will be cropped with `self.params.crop_slice_tuple` if it is not None.
        """
        image2_slice = image2
        if self.params.crop_slice_tuple is not None:
            image2_slice = image2[self.params.crop_slice_tuple]
            center = np.array([slice_.start + (slice_.stop - slice_.start)*0.5 for slice_ in self.params.crop_slice_tuple])
        else:
            center = np.array(self.meta.dataset_shape.sig) * 0.5
        ccorr = normalized_corr(np.atleast_2d(image1), np.atleast_2d(image2_slice))
        error, maximum, max_pos = find_ccorr_max(ccorr)
        if error:
            logging.error('Maximum not found.')
            return
        if maximum < self.params.ccorr_threshold:
            logging.error(f'Cross-correlation coefficient below threshold ({maximum:.3g} < {self.params.ccorr_threshold}).')
            return
        
        return np.array(max_pos) - center
    
    def preprocess(self):
        self.results.processed_frame_indices[:] = -1
        
    def process_frame(self, frame):
        max_pos = self.calculate_shift(self.params.reference_frame, frame)
        if max_pos is None:
            return
        self.results.processed_frame_indices[self.meta.slice.origin[0]] = self.meta.slice.origin[0]
        self.results.shifts_buffer[:] = max_pos
        shifted = shift(frame, max_pos, order=1, cval=np.mean(frame))
        self.results.sum_buffer[:] += shifted
        self.results.number_processed_frames[:] += 1
        
    def merge(self, dest, src):
#        if dest['number_processed_frames']:
#            max_pos = self.calculate_shift(dest['sum_buffer'], src['sum_buffer'])
#            if max_pos is None:
#                return
#            max_pos = np.array(max_pos) - np.array(self.meta.dataset_shape.sig) * 0.5
#            flat_shift_buffer = src['shifts_buffer'].ravel()
#            indices = src['processed_frame_indices']
#            flat_shift_buffer[indices[indices > -1]] += max_pos
#            shifted = shift(src['sum_buffer'], max_pos, order=1, cval=np.mean(src['sum_buffer']))
#            dest['sum_buffer'][:] += src['sum_buffer']
#        else:
        dest['sum_buffer'][:] += src['sum_buffer']
        dest['shifts_buffer'][:] = src['shifts_buffer']
        dest['number_processed_frames'][:] += src['number_processed_frames']
        

class AlignedAverage:
    def __init__(self, computation, **kwargs):
        self.computation = computation
        self.__api = computation.api
        self.__event_loop = self.__api.application.document_controllers[0]._document_controller.event_loop
        self.__objects_to_close = list()
        
        def close_ui():
            for obj in self.__objects_to_close:
                obj.unbind_text()
            self.__objects_to_close = list()
            
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
            
            x_index_variable = self.computation._computation._get_variable('reference_frame_index_1')
            y_index_variable = self.computation._computation._get_variable('reference_frame_index_0')
            column = ui.create_column_widget()
            row = ui.create_row_widget()

            label = ui.create_label_widget('Reference frame index (x, y):')
            x_index_line = ui.create_line_edit_widget()
            y_index_line = ui.create_line_edit_widget()
            row.add_spacing(10)
            row.add(label)
            row.add_spacing(5)
            row.add(x_index_line)
            row.add_spacing(5)
            row.add(y_index_line)
            row.add_stretch()
            row.add_spacing(10)

            column.add_spacing(10)
            column.add(row)
            column.add_spacing(10)
            column.add_stretch()
            
            if x_index_variable:
                x_index_line._widget.bind_text(Binding.PropertyBinding(x_index_variable, 'value', converter=Converter.IntegerToStringConverter()))
            if y_index_variable:
                y_index_line._widget.bind_text(Binding.PropertyBinding(y_index_variable, 'value', converter=Converter.IntegerToStringConverter()))
            
            self.__objects_to_close.append(x_index_line._widget)
            self.__objects_to_close.append(y_index_line._widget)

            return column

        self.computation._computation.create_panel_widget = create_panel_widget
        self.computation._computation.close_ui = close_ui
        
    def get_xdata_for_results(self, result_array, is_sequence=False):
        ndim = result_array.ndim - int(is_sequence)
        data_descriptor = self.__api.create_data_descriptor(is_sequence, 0, ndim)
        intensity_calibration = self.__api.create_calibration()
        xdata = self.__api.create_data_and_metadata(result_array,
                                                    intensity_calibration=intensity_calibration,
                                                    data_descriptor=data_descriptor)
        return xdata
        
    async def run_udf(self, udf: UDF, cancel_id, executor, dataset, roi=None):
        try:
            result_iter = UDFRunner(udf).run_for_dataset_async(
                dataset, executor, roi=roi, cancel_id=cancel_id,
            )
    
            async for result in result_iter:
                num_frames = result['number_processed_frames'].data[0]
                result_array = np.array(result['sum_buffer']) / num_frames
                self.__new_xdata_list = [self.get_xdata_for_results(result_array)]
                result_array = np.moveaxis(np.array(result['shifts_buffer']), -1, 0)
                self.__new_xdata_list.append(self.get_xdata_for_results(result_array, is_sequence=True))
                self.commit()
        except JobCancelledError:
            pass
            
    def execute(self, src, align_region, reference_frame_index_0, reference_frame_index_1):
        try:
            continue_ = False
            if align_region:
                align_region = align_region[0]
            if hasattr(self.computation._computation, 'last_src_uuid'):
                if str(src.uuid) != self.computation._computation.last_src_uuid:
                    continue_ = True
                    if hasattr(self.computation._computation, 'ds'):
                        self.computation._computation.ds = None
            else:
                continue_ = True
            if hasattr(self.computation._computation, 'last_align_region'):        
                if align_region.persistent_dict != self.computation._computation.last_align_region:
                    continue_ = True
            else:
                continue_ = True
            if hasattr(self.computation._computation, 'last_reference_frame_index'):
                if (reference_frame_index_0, reference_frame_index_1) != self.computation._computation.last_reference_frame_index:
                    continue_ = True
            else:
                continue_ = True
            if not continue_:
                return
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
            if align_region:
                if align_region.type == 'rect-graphic':
                    top = round(align_region.bounds[0][0] * shape[0])
                    left = round(align_region.bounds[0][1] * shape[1])
                    height = round(align_region.bounds[1][0] * shape[0])
                    width = round(align_region.bounds[1][1] * shape[1])
                    crop_slice_tuple = (slice(top, top + height), slice(left, left + width))
                elif align_region.type == 'interval-graphic':
                    crop_slice_tuple = (slice(round(align_region.interval[0] * shape[0]), round(align_region.interval[1] * shape[0])), )
            else:
                crop_slice_tuple = None
            if hasattr(self.computation._computation, 'ds') and self.computation._computation.ds:
                ds = self.computation._computation.ds
            else:
                ds = dataset.load(file_type, executor.ensure_sync(), **file_parameters)
                self.computation._computation.ds = ds
                
            if len(ds.shape.nav) == 2:
                reference_frame_index = (reference_frame_index_0, reference_frame_index_1)
            else:
                reference_frame_index = (reference_frame_index_0, )
                
            if src.xdata.is_collection or src.xdata.is_sequence:
                reference_frame = src.xdata.data[reference_frame_index]
            else:
                roi = np.zeros(ds.shape.nav, dtype=bool)
                roi[reference_frame_index] = 1
                result = UDFRunner(PickUDF()).run_for_dataset(ds, executor.ensure_sync(), roi=roi)
                reference_frame = np.squeeze(np.array(result['intensity']))
            
            udf = AlignedSumUDF(reference_frame, crop_slice_tuple=crop_slice_tuple)
            dc = self.__api.application.document_controllers[0]._document_controller
            if hasattr(self.computation._computation, 'cancel_id'):
                to_cancel = self.computation._computation.cancel_id
                self.__api.queue_task(lambda: self.__event_loop.create_task(executor.cancel(to_cancel)))
            self.computation._computation.cancel_id = str(time.time())
            dc.add_task('libertem-aligend_average', lambda: self.__event_loop.create_task(self.run_udf(udf, self.computation._computation.cancel_id, executor, dataset=ds)))
            
            self.computation._computation.last_src_uuid = str(src.uuid)
            if align_region:
                self.computation._computation.last_align_region = copy.deepcopy(align_region.persistent_dict)
            self.computation._computation.last_reference_frame_index = (reference_frame_index_0, reference_frame_index_1)
            
        except Exception:
            import traceback
            traceback.print_exc()

    def commit(self):
        try:
            self.computation.set_referenced_xdata('average_data_item', self.__new_xdata_list[0])
            self.computation.set_referenced_xdata('shifts_data_item', self.__new_xdata_list[1])
        except AttributeError:
            pass


class AlignedAverageMenuItem:

    menu_id = 'libertem_menu'  # required, specify menu_id where this item will go
    menu_name = _('LiberTEM') # optional, specify default name if not a standard menu
    menu_before_id = 'window_menu' # optional, specify before menu_id if not a standard menu
    menu_item_name = _('Aligned Average')  # menu item name

    def __init__(self, api):
        self.__api = api
        def init():
            document_controller = self.__api.application.document_controllers[0]._document_controller
            computation_data_items = document_controller.ui.get_persistent_string('libertem_aligned_average_data_items_0')
            self.__computation_data_items = json.loads(computation_data_items) if computation_data_items else dict()
            self.__tool_tip_boxes = list()
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
            return
            text = ('Click "Select" in the computation panel (Window -> Computation) to update the masks used for this '
                    'computation. All graphics tagged as mask (Processing -> Graphics -> Add to Mask) will be used.'
                    '\nWithout a mask, the whole frames will be summed.')
        elif tip_id == 'map_4d':
            return
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
            shifts_data_item = self.__api.library.create_data_item(title='Shifts of ' + data_item.title)
            average_data_item = self.__api.library.create_data_item(title='Aligned average of ' + data_item.title)
            display_item = document_controller.document_model.get_display_item_for_data_item(shifts_data_item._data_item)
            show_display_item(window, display_item)
            display_item = document_controller.document_model.get_display_item_for_data_item(average_data_item._data_item)
            show_display_item(window, display_item)
            align_region = list()
            
            selection_data_item = None
            
            if api_data_item.xdata.collection_dimension_count == 2 and api_data_item.xdata.datum_dimension_count == 1:
                for computation in self.__api.application._application.document_model.computations:
                     if computation.processing_id == 'pick-point' and data_item in computation._inputs:
                         selection_data_item = Facade.DataItem(computation.get_output('target'))
                         break
            elif (api_data_item.xdata.is_sequence or api_data_item.xdata.collection_dimension_count == 1) and api_data_item.xdata.datum_dimension_count in {1, 2}:
                selection_data_item = api_data_item
            
            if selection_data_item:
                for graphic in selection_data_item.graphics:
                    if graphic._graphic.role == 'mask' and graphic.graphic_type in {'rect-graphic', 'interval-graphic'}:
                        align_region.append(graphic)
                        break
                 
            computation = self.__api.library.create_computation('nion.libertem.aligned_average',
                                                                inputs={'src': api_data_item,
                                                                        'align_region': align_region,
                                                                        'reference_frame_index_0': 0,
                                                                        'reference_frame_index_1': 0},
                                                                outputs={'shifts_data_item': shifts_data_item,
                                                                         'average_data_item': average_data_item})
            computation._computation.source = data_item
            if ds is not None:
                computation._computation.ds = ds
                x_index_variable = computation._computation._get_variable('reference_frame_index_1')
                y_index_variable = computation._computation._get_variable('reference_frame_index_0')
                x_index_variable.value_min = 0
                y_index_variable.value_min = 0
                if len(ds.shape.nav) > 1:
                    x_index_variable.value_max = ds.shape.nav[1] - 1
                    y_index_variable.value_max = ds.shape.nav[0] - 1
                else:
                    x_index_variable.value_max = ds.shape.nav[0] - 1
                    y_index_variable.value_max = 0
              
            self.__computation_data_items.update({str(data_item.uuid): 'source',
                                                  str(shifts_data_item._data_item.uuid): 'shifts_data_item',
                                                  str(average_data_item._data_item.uuid): 'average_data_item'})
            self.__api.application.document_controllers[0]._document_controller.ui.set_persistent_string('libertem_aligend_average_data_items_0', json.dumps(self.__computation_data_items))
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
        self.__menu_item_ref = api.create_menu_item(AlignedAverageMenuItem(api))

    def close(self):
        self.__menu_item_ref.close()

Symbolic.register_computation_type('nion.libertem.aligned_average', AlignedAverage)
