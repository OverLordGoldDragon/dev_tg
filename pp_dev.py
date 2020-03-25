"""TODO:
    - variable windows_per_batch support
"""

class TimeseriesPreprocessor():
    # `slice_idx` == `window`
    # `slices_per_batch` == `windows_per_batch`

    def __init__(self, window_size, batch_timesteps,
                 slide_size=None,
                 start_increments=None,
                 loadskip_list=None):
        self.window_size=window_size
        self.batch_timesteps=batch_timesteps
        self.slide_size=slide_size or window_size
        self.start_increments=start_increments
        
        self.start_increment = 0
        self.reset_state()
        self._set_start_increment(epoch=0)
        self._set_windows_per_batch()
        
        self.loadskip_list=loadskip_list or [
            'start_increments', 'window_size', 'slide_size']

    def process(self, batch):
        return self._next_window(batch)

    def _next_window(self, batch):
        start = self.slice_idx * self.slide_size + self.start_increment
        end   = start + self.window_size
        return batch[:, start:end]

    def reset_state(self):
        self.slice_idx = 0
        self.batch_exhausted = True
        self.batch_loaded = False

    def on_epoch_end(self, epoch):
        self._set_start_increment(epoch)
        self._set_windows_per_batch()

    def update_state(self):
        self.slice_idx += 1
        if self.slice_idx == self.slices_per_batch:
            self.batch_exhausted = True
            self.batch_loaded = False

    ##################################
    def _set_start_increment(self, epoch):
        if self.start_increments is not None:
            self.start_increment = self.start_increments[
                     epoch % len(self.start_increments)]

    def _set_windows_per_batch(self):
        self.slices_per_batch = 1 + (
            self.batch_timesteps - self.window_size - self.start_increment
            ) // self.slide_size
        self.slices_per_batch = self.slices_per_batch

class GenericPreprocessor():
    def __init__(self, loadskip_list=None):        
        self.loadskip_list=loadskip_list or []
        self.reset_state()

    def process(self, batch):
        return batch

    def reset_state(self):
        self.batch_exhausted = True
        self.batch_loaded = False

    def on_epoch_end(self, epoch):
        pass

    def update_state(self):
        self.reset_state()
