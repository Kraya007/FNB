import sklearn.utils.validation
import sklearn.utils

_original_check_array = sklearn.utils.validation.check_array

def patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _original_check_array(*args, **kwargs)

sklearn.utils.validation.check_array = patched_check_array
sklearn.utils.check_array = patched_check_array

