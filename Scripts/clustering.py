import numpy as np
from skimage.feature import peak_local_max
import scipy.sparse
import sparse
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from libertem.udf.stddev import run_stddev
from libertem.udf.sum import SumUDF
from libertem.udf.masks import ApplyMasksUDF

from nion.typeshed import Interactive_1_0 as Interactive
from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UI

import utils.interface as iface


def clustering(interactive: Interactive, api: API):
    window = api.application.document_windows[0]
    target_data_item = window.target_data_item
    ctx = iface.get_context()
    ds = iface.dataset_from_data_item(ctx, target_data_item)
    fy, fx = tuple(ds.shape.sig)
    y, x = tuple(ds.shape.nav)
    # roi = np.random.choice([True, False], tuple(ds.shape.nav), p=[0.01, 0.99])
    # We only sample 5 % of the frame for the std deviation map
    # since the UDF still needs optimization
    std_roi = np.random.choice([True, False], tuple(ds.shape.nav), p=[0.05, 0.95])
    roi = np.ones((y, x), dtype=bool)
    # roi = np.zeros((y, x), dtype=bool)
    # roi[:, :50] = True
    stddev_res = run_stddev(ctx=ctx, dataset=ds, roi=std_roi*roi)
    ref_frame = stddev_res['std']
    # sum_res = ctx.run_udf(udf=SumUDF(), dataset=ds)
    # ref_frame = sum_res['intensity'].data
    update_data(target_data_item, ref_frame)

    peaks = peak_local_max(ref_frame, min_distance=3, num_peaks=500)
    masks = sparse.COO(
        shape=(len(peaks), fy, fx),
        coords=(range(len(peaks)), peaks[..., 0], peaks[..., 1]),
        data=1
    )
    feature_udf = ApplyMasksUDF(
        mask_factories=lambda: masks,
        mask_dtype=np.uint8,
        mask_count=len(peaks),
        use_sparse=True
    )
    feature_res = ctx.run_udf(udf=feature_udf, dataset=ds, roi=roi)
    f = feature_res['intensity'].raw_data.astype(np.float32)
    f = np.log(f - np.min(f) + 1)
    feature_vector = f / np.abs(f).mean(axis=0)
    # too slow
    # nion_peaks = peaks / tuple(ds.shape.sig)
    # with api.library.data_ref_for_data_item(target_data_item):    
    #     for p in nion_peaks:
    #         target_data_item.add_ellipse_region(*p, 0.01, 0.01)
    connectivity = scipy.sparse.csc_matrix(
        grid_to_graph(
            # Transposed!
            n_x=y,
            n_y=x,
        )
    )

    roi_connectivity = connectivity[roi.flatten()][:, roi.flatten()]
    threshold = interactive.get_float("Cluster distance threshold: ", 10)
    clusterer = AgglomerativeClustering(
        affinity='euclidean',
        distance_threshold=threshold,
        n_clusters=None,
        linkage='ward',
        connectivity=roi_connectivity,
    )
    clusterer.fit(feature_vector)
    labels = np.zeros((y, x), dtype=np.int32)
    labels[roi] = clusterer.labels_ + 1
    new_data = api.library.create_data_item_from_data(labels)
    window.display_data_item(new_data)


def update_data(data_item, data):
    metadata = iface.convert_from_facade(data_item.metadata)
    dimensional_calibrations = list(data_item.dimensional_calibrations)
    intensity_calibration = data_item.intensity_calibration
    data_item.data = data
    data_item.set_metadata(metadata)
    data_item.set_dimensional_calibrations(dimensional_calibrations)
    data_item.set_intensity_calibration(intensity_calibration)


def script_main(api_broker):
    # type: Interactive
    interactive = api_broker.get_interactive(Interactive.version)
    api = api_broker.get_api(API.version, UI.version)  # type: API
    clustering(interactive, api)
