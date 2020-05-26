import builtins
import numpy as np
from functools import reduce
from pprint import pprint
from collections.abc import Mapping


def ordered_shuffle(*args):
    zipped_args = list(zip(*(a.items() if isinstance(a, dict)
                             else a for a in args)))
    np.random.shuffle(zipped_args)
    return [(_type(data) if _type != np.ndarray else np.asarray(data))
            for _type, data in zip(map(type, args), zip(*zipped_args))]


def nCk(n, k):  # n-Choose-k
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom


def builtin_or_npscalar(x):
    return (type(x) in (*vars(builtins).values(), type(None), type(min)) or
            isinstance(x, np.generic))


def deeplen(item):
    if isinstance(item, np.ndarray):
        return item.size
    try:
        list(iter(item))
    except:
        return 1
    if isinstance(item, str):
        return 1
    if isinstance(item, Mapping):
        item = item.values()
    return sum(deeplen(subitem) for subitem in item)


def deepget(obj, key=None, drop_keys=0):
    if not key or not obj:
        return obj
    if drop_keys != 0:
        key = key[:-drop_keys]
    for k in key:
        if isinstance(obj, Mapping):
            try:
                k = list(obj)[k]  # get key by index (OrderedDict, Python >=3.6)
            except:
                print('\n')
                print(obj)
                print(key)
                print(len(obj))
                raise Exception
        obj = obj[k]
    return obj


def deepmap(obj, fn):
    def dkey(x, k):
        return list(x)[k] if isinstance(x, Mapping) else k

    def nonempty_iter(item):
        # do not enter empty iterable, since nothing to 'iterate' or apply fn to
        try:
            list(iter(item))
        except:
            return False
        return not isinstance(item, str) and len(item) > 0

    def _process_key(obj, key, depth, revert_tuple_keys, recursive=False):
        container = deepget(obj, key, 1)
        item      = deepget(obj, key, 0)

        if nonempty_iter(item) and not recursive:
            depth += 1
        if len(key) == depth:
            if key[-1] == len(container) - 1:  # iterable end reached
                depth -= 1      # exit iterable
                key = key[:-1]  # drop iterable key
                if key in revert_tuple_keys:
                    supercontainer = deepget(obj, key, 1)
                    k = dkey(supercontainer, key[-1])
                    supercontainer[k] = tuple(deepget(obj, key))
                    revert_tuple_keys.pop(revert_tuple_keys.index(key))
                if depth == 0 or len(key) == 0:
                    key = None  # exit flag
                else:
                    # recursively exit iterables, decrementing depth
                    # and dropping last key with each recursion
                    key, depth = _process_key(obj, key, depth, revert_tuple_keys,
                                              recursive=True)
            else:  # iterate next element
                key[-1] += 1
        elif depth > len(key):
            key.append(0)  # iterable entry
        return key, depth

    key = [0]
    depth = 1
    revert_tuple_keys = []

    if not nonempty_iter(obj):  # nothing to do here
        raise ValueError(f"input must be a non-empty iterable - got: {obj}")
    if isinstance(obj, tuple):
        obj = list(obj)
        revert_tuple_keys.append(None)  # revert to tuple at function exit

    while key is not None:
        container = deepget(obj, key, 1)
        item      = deepget(obj, key, 0)

        if isinstance(container, tuple):
            ls = list(container)  # cast to list to enable mutating
            ls[key[-1]] = fn(item, key)

            supercontainer = deepget(obj, key, 2)
            k = dkey(supercontainer, key[-2])
            supercontainer[k] = ls
            revert_tuple_keys.append(key[:-1])  # revert to tuple at iterable exit
        else:
            k = dkey(container, key[-1])
            container[k] = fn(item, key)

        key, depth = _process_key(obj, key, depth, revert_tuple_keys)

    if None in revert_tuple_keys:
        obj = tuple(obj)
    return obj


# TODO cleanup
def deepcopy_v2(obj, item_fn=None, skip_flag=42069, debug_verbose=False):
    """Enables customized copying of a nested iterable, mediated by `item_fn`."""
    if item_fn is None:
        item_fn = lambda item: item
    copied = [] if isinstance(obj, (list, tuple)) else {}
    copied_key = []
    revert_tuple_keys = []
    copy_paused = [False]
    key_decrements = [0]
    skipref_key = []

    def dkey(x, k):
        return list(x)[k] if isinstance(x, Mapping) else k

    def isiter(item):
        try:
            list(iter(item))
            return not isinstance(item, str)
        except:
            return False

    def reconstruct(item, key):
        def _container_or_elem(item):
            if isiter(item):
                if isinstance(item, (tuple, list)):
                    return []
                elif isinstance(item, Mapping):
                    return {}
            return item_fn(item)

        def _obj_key_advanced(key, skipref_key):
            # [1, 1]    [1, 0] -> True
            # [2]       [1, 0] -> True
            # [2, 0]    [1, 0] -> True
            # [1, 0]    [1, 0] -> False
            # [1]       [1, 0] -> False
            # [1, 0, 1] [1, 0] -> False
            i = 0
            while (i < len(key) - 1 and i < len(skipref_key) - 1) and (
                    key[i] == skipref_key[i]):
                i += 1
            return key[i] > skipref_key[i]

        def _update_copied_key(key, copied_key, on_skip=False):
            ck = []
            for i, (k, k_decrement) in enumerate(zip(key, key_decrements)):
                ck.append(k - k_decrement)
            copied_key[:] = ck

        def _update_key_decrements(key, key_decrements, on_skip=False):
            if on_skip:
                while len(key_decrements) < len(key):
                    key_decrements.append(0)
                while len(key_decrements) > len(key):
                    key_decrements.pop()
                key_decrements[len(key) - 1] += 1
            else:
                while len(key_decrements) < len(key):
                    key_decrements.append(0)
                while len(key_decrements) > len(key):
                    key_decrements.pop()

        def _copy(obj, copied, key, copied_key, _item):
            container = deepget(copied, copied_key, 1)

            if isinstance(container, list):
                container.insert(copied_key[-1], _item)
            elif isinstance(container, str):
                # str container implies container was transformed to str by
                # item_fn; continue skipping until deepmap exits container in obj
                pass
            else:  # tuple will yield error, no need to catch
                obj_container = deepget(obj, key, 1)
                k = dkey(obj_container, key[-1])
                if debug_verbose:
                    print("OBJ_CONTAINER:", obj_container, key[-1])
                    print("CONTAINER:", container, k, '\n')
                container[k] = _item

        if copy_paused[0] and not _obj_key_advanced(key, skipref_key):
            if debug_verbose:
                print(">SKIP:", item)
            return item


        _item = _container_or_elem(item)
        if isinstance(_item, int) and _item == skip_flag:
            copy_paused[0] = True
            _update_key_decrements(key, key_decrements, on_skip=True)
            skipref_key[:] = key
            if debug_verbose:
                print("SKIP:", key, key_decrements, copied_key)
                print(item)
            return item
        copy_paused[0] = False

        _update_key_decrements(key, key_decrements)
        while len(key_decrements) > len(key):
            key_decrements.pop()
            if debug_verbose:
                pprint("POP: {} {} {}".format(key, key_decrements, copied_key))
        _update_copied_key(key, copied_key)

        if debug_verbose:
            print("\nSTUFF:", key, key_decrements, copied_key, len(copied))
            print(_item)
            for k, v in copied.items():
                print(k, '--', v)
            print()
        _copy(obj, copied, key, copied_key, _item)
        if debug_verbose:
            print("###########################################################",
                  len(copied))

        if isinstance(item, tuple):
            revert_tuple_keys.append(copied_key.copy())
        return item

    def _revert_tuples(copied, obj, revert_tuple_keys):
        revert_tuple_keys = list(reversed(sorted(revert_tuple_keys,
                                                 key=lambda x: len(x))))
        for key in revert_tuple_keys:
            supercontainer = deepget(copied, key, 1)
            container      = deepget(copied, key, 0)
            k = dkey(supercontainer, key[-1])
            supercontainer[k] = tuple(container)
        if isinstance(obj, tuple):
            copied = tuple(copied)
        return copied

    deepmap(obj, reconstruct)
    copied = _revert_tuples(copied, obj, revert_tuple_keys)
    return copied


def deep_isinstance(obj, cond):
    bools = []
    def fn(item, key=None):
        if isinstance(item, str):
            bools.append(cond(item))
            return item
        try:
            list(iter(item))
        except TypeError:
            bools.append(cond(item))
        return item

    try:
        list(iter(obj))
        assert len(obj) > 0
        deepmap(obj, fn)
    except:
        fn(obj)
    return bools
