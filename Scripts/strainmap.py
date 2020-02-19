import numpy as np
from libertem.udf.logsum import LogsumUDF
from libertem.analysis.fullmatch import FullMatcher
from libertem_blobfinder.common.patterns import BackgroundSubtraction
from libertem_blobfinder.common.correlation import get_peaks
from libertem_blobfinder.udf.refinement import run_refine

from nion.typeshed import Interactive_1_0 as Interactive
from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UI

import utils.interface as iface


def strainmap(interactive: Interactive, api: API):
    window = api.application.document_windows[0]
    target_data_item = window.target_data_item
    ctx = iface.get_context()
    ds = iface.dataset_from_data_item(ctx, target_data_item)
    c = find_center_mask(target_data_item)
    if c is None:
        add_center_mask(interactive, target_data_item)
        return
    size = np.array(tuple(c.size)) * tuple(ds.shape.sig)
    radius = np.mean(size) / 2
    logsum = ctx.run_udf(udf=LogsumUDF(), dataset=ds)
    update_data(target_data_item, logsum['logsum'].data)
    pattern = BackgroundSubtraction(radius=radius, radius_outer=2*radius)
    peaks = get_peaks(logsum['logsum'].data, pattern, 15)
    nion_peaks = peaks / tuple(ds.shape.sig)
    nion_radius = radius / tuple(ds.shape.sig) * 2
    for p in nion_peaks:
        target_data_item.add_ellipse_region(*p, *nion_radius)
    matcher = FullMatcher()
    (matches, unmatched, weak) = matcher.full_match(peaks)
    match = matches[0]
    match, indices = run_refine(
        ctx=ctx,
        dataset=ds,
        zero=match.zero,
        a=match.a,
        b=match.b,
        match_pattern=pattern,
        matcher=matcher,
    )
    a_b = np.linalg.norm(match['a'], axis=-1) / np.linalg.norm(match['b'], axis=-1)
    # Eliminate NaN values that mess up plotting in Nion Swift
    a_b[np.isnan(a_b)] = 1
    new_data = api.library.create_data_item_from_data(a_b)
    window.display_data_item(new_data)


def find_center_mask(data_item):
    for g in data_item.graphics:
        if g.type == 'ellipse-region' and g._item.role == 'mask':
            return g
    return None


def update_data(data_item, data):
    metadata = iface.convert_from_facade(data_item.metadata)
    dimensional_calibrations = list(data_item.dimensional_calibrations)
    intensity_calibration = data_item.intensity_calibration
    data_item.data = data
    data_item.set_metadata(metadata)
    data_item.set_dimensional_calibrations(dimensional_calibrations)
    data_item.set_intensity_calibration(intensity_calibration)


def add_center_mask(interactive: Interactive, data_item):
    e = data_item.add_ellipse_region(0.5, 0.5, 0.1, 0.1)
    e._item.role = 'mask'
    interactive.alert("Adjust mask to zero order peak and run again")


def script_main(api_broker):
    interactive = api_broker.get_interactive(Interactive.version)  # type: Interactive
    api = api_broker.get_api(API.version, UI.version)  # type: API
    strainmap(interactive, api)
