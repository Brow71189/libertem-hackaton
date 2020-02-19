from nion.utils import Registry
from libertem import api
from libertem.executor.inline import InlineJobExecutor


def convert_from_facade(x):
    if hasattr(x, '__next__'):
        return [convert_from_facade(xx) for xx in x]
    if isinstance(x, list):
        return [convert_from_facade(xx) for xx in x]
    if isinstance(x, tuple):
        return (convert_from_facade(xx) for xx in x)
    if isinstance(x, dict):
        return {k: convert_from_facade(v) for k, v in x.items()}
    return x


def dataset_from_data_item(ctx, data_item):
    metadata = convert_from_facade(data_item.metadata)
    params = metadata['libertem-io']['file_parameters']
    params['filetype'] = params.pop('type')
    return ctx.load(**params)


def get_context():
    executor = Registry.get_component('libertem_executor')
    return api.Context(executor=executor.ensure_sync())


def get_inline_context():
    return api.Context(executor=InlineJobExecutor())
