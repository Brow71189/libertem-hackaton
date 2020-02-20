from nion.utils import Registry
from libertem import api
from libertem.executor.inline import InlineJobExecutor


def dataset_from_data_item(ctx, data_item):
    metadata = data_item.metadata
    params = metadata['libertem-io']['file_parameters']
    params['filetype'] = params.pop('type')
    return ctx.load(**params)


def get_context():
    executor = Registry.get_component('libertem_executor')
    return api.Context(executor=executor.ensure_sync())


def get_inline_context():
    return api.Context(executor=InlineJobExecutor())
