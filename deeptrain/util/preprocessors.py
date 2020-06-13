from abc import ABCMeta, abstractmethod
from ._backend import WARN


class Preprocessor(metaclass=ABCMeta):
    """Abstract base class for preprocessors, outlining required methods for
    operability with :class:`DataGenerator`.

    The following attributes are "synched" with :class:`DataGenerator`:
    `batch_loaded`, `batch_exhausted`, `slices_per_batch`, `slice_idx`. Setter
    and getter are implemented to set and get these attributes from the
    preprocessor, so they are always same for `Preprocessor` and `DataGenerator`.
    """
    def __new__(cls, *args, **kwargs):
        """Ensures preprocessors have expected attributes defined.
        """
        cls.batch_loaded = False
        cls.batch_exhausted = False
        cls.slices_per_batch = None
        cls.slice_idx = None
        return object.__new__(cls)

    @abstractmethod
    def process(self, batch, labels):
        """Required to implement; must return `(batch, labels)`. Can
        apply arbitrary preprocessing steps, or return as-is.
        Is called within :meth:`DataGenerator.get`.
        """
        pass

    @abstractmethod
    def update_state(self):
        """Required to implement; must set `batch_exhausted` and `batch_loaded`
        attributes to True or False.
        """
        pass

    def reset_state(self):
        """Optional to implement. Can be used to reset attributes specific
        to the preprocessor.
        Is called within :meth:`DataGenerator.reset_state`.
        """
        pass

    def on_epoch_end(self, epoch):
        """Optional to implement. Can be used to do things at end of epoch.
        Is called within :meth:`DataGenerator.on_epoch_end`, which is
        called by `_on_epoch_end` within
        :meth:`TrainGenerator._train_postiter_processing` or
        :meth:`TrainGenerator._val_postiter_processing`.
        """
        pass

    def on_init_end(self, batch, labels, probing=False):
        """Optional to implement. Can be used to set data-specific attributes
        before training begins.
        Is called at the end of `DataGenerator.__init__()`, where `batch`
        and `labels` are fed without advancing any internal counters.

        `DataGenerator` will first call this method with `probing=True` to see
        if it's been implemented; by default, not implemented returns False.
        Implementations must hence set `if probing: return True`. This is to
        avoid redundantly calling :meth:`DataGenerator._get_next_batch`, which
        is a potentially expensive operation.
        """
        if probing:
            return False

    def _validate_configs(self):
        """Internal method to validate `slices_per_batch` in
        :meth:`DataGenerator._set_preprocessor`.
        """
        spb = self.slices_per_batch
        assert (spb is None or (isinstance(spb, int) and spb >= 1)
                ), ("`slices_per_batch` must be None or int >= 1, got: %s" % spb)


class TimeseriesPreprocessor(Preprocessor):
    """Stateful preprocessor breaking up batches into "windows".

    Arguments:
        window_size: int
            Length of each window (dim1), or number of timesteps per slice.
        batch_timesteps: int


    A "slice" here is a "window", and `slices_per_batch` is the number of such
    windows per batch. `

    If `batch_shape == (32, 100, 4)`, and `window_size == 25`,
    then `slices_per_batch == 4` (100 / 25), assuming `slide_size == window_size`.
    """
    def __init__(self, window_size,
                 slide_size=None,
                 start_increments=None,
                 loadskip_list=None):
        self.window_size=window_size
        self.slide_size=slide_size or window_size
        self.start_increments=start_increments

        self._start_increment = 0
        self.reset_state()
        if start_increments is not None:
            self._set_start_increment(epoch=0)

        self.loadskip_list=loadskip_list or [
            'start_increments', 'window_size', 'slide_size']

    def process(self, batch, labels):
        return self._next_window(batch), labels

    def _next_window(self, batch):
        start = self.slice_idx * self.slide_size + self.start_increment
        end   = start + self.window_size
        return batch[:, start:end]

    def reset_state(self):
        self.slice_idx = 0

    def on_epoch_end(self, epoch):
        if self.start_increments is not None:
            self._set_start_increment(epoch)
        self._set_slices_per_batch()

    def update_state(self):
        self.slice_idx += 1
        if self.slices_per_batch is None:  # is the case before any data is fed
            # enables determining timesteps automatically at runtime
            self._set_slices_per_batch()
        if self.slice_idx == self.slices_per_batch:
            self.batch_exhausted = True
            self.batch_loaded = False

    def on_init_end(self, batch, labels, probing=False):
        if probing:
            return True  # signal DataGenerator that method is implemented
        self.batch_timesteps = batch.shape[1]
        self._set_slices_per_batch()

    #### Helper methods #######################################################
    def _set_slices_per_batch(self):
        self.slices_per_batch = 1 + (
            self.batch_timesteps - self.window_size - self.start_increment
            ) // self.slide_size

    def _set_start_increment(self, epoch):
        self.start_increment = self.start_increments[
            epoch % len(self.start_increments)]

    @property
    def start_increment(self):
        return self._start_increment

    @start_increment.setter
    def start_increment(self, value):
        def _validate(value):
            assert isinstance(value, int), ("`start_increment` must be set to "
                                            "integer (got: %s)" % value)
            if value not in self.start_increments:
                print(WARN, ("setting `start_increment` to {}, which is not in "
                             "`start_increments` ({})").format(
                                 value, ", ".join(self.start_increments)))

        if self.start_increments is not None:
            _validate(value)
            self._start_increment = value
        else:
            raise ValueError("setting `start_increment` is unsupported if "
                             "`start_increments` is None")


class GenericPreprocessor(Preprocessor):
    def __init__(self, loadskip_list=None):
        self.loadskip_list=loadskip_list or []
        self.reset_state()

    def process(self, batch, labels):
        return batch, labels

    def reset_state(self):
        self.batch_exhausted = True
        self.batch_loaded = False

    def update_state(self):
        self.reset_state()
