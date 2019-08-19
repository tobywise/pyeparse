# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
from copy import deepcopy

from ._event import find_events
from ._fixes import string_types
from .viz import plot_calibration, plot_heatmap_raw, plot_raw
from .utils import _filt_update_info
from mne.filter import filter_data


class _BaseRaw(object):
    """Base class for Raw"""
    def __init__(self):
        assert self._samples.shape[0] == len(self.info['sample_fields'])
        assert self.times[0] == 0.0
        assert isinstance(self.info['sfreq'], float)
        dt = np.abs(np.diff(self.times) - (1. / self.info['sfreq']))
        assert np.all(dt < 1e-6)

    def __repr__(self):
        return '<Raw | {0} samples>'.format(self.n_samples)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            idx = (idx,)
        elif isinstance(idx, slice):
            idx = (idx,)
        if not isinstance(idx, tuple):
            raise TypeError('index must be a string, slice, or tuple')
        if isinstance(idx[0], string_types):
            idx = list(idx)
            idx[0] = self._di(idx[0])
            idx = tuple(idx)
        if len(idx) > 2:
            raise ValueError('indices must have at most two elements')
        elif len(idx) == 1:
            idx = (idx[0], slice(None))
        data = self._samples[idx]
        times = self.times[idx[1:]]
        return data, times

    def _di(self, key):
        """Helper to get the sample dict index"""
        if key not in self.info['sample_fields']:
            raise KeyError('key "%s" not in sample fields %s'
                           % (key, self.info['sample_fields']))
        return self.info['sample_fields'].index(key)

    def save(self, fname, overwrite=False):
        """Save data to HD5 format

        Parameters
        ----------
        fname : str
            Filename to use.
        overwrite : bool
            If True, overwrite file (if it exists).
        """
        if op.isfile(fname) and not overwrite:
            raise IOError('file "%s" exists, use overwrite=True to overwrite'
                          % fname)
        try:
            import h5py
        except Exception:
            raise ImportError('h5py could not be imported')
        with h5py.File(fname, mode='w') as fid:
            # samples
            comp_kw = dict(compression='gzip', compression_opts=5)
            s = np.core.records.fromarrays(self._samples)
            s.dtype.names = self.info['sample_fields']
            fid.create_dataset('samples', data=s, **comp_kw)
            # times
            fid.create_dataset('times', data=self._times, **comp_kw)
            # discrete
            dg = fid.create_group('discrete')
            for key, val in self.discrete.items():
                dg.create_dataset(key, data=val, **comp_kw)
            # info (harder)
            info = deepcopy(self.info)
            info['meas_date'] = info['meas_date'].isoformat()
            items = [('eye', '|S256'),
                     ('camera', '|S256'),
                     ('camera_config', '|S256'),
                     ('meas_date', '|S32'),
                     ('ps_units', '|S16'),
                     ('screen_coords', 'f8', self.info['screen_coords'].shape),
                     ('serial', '|S256'),
                     ('sfreq', 'f8'),
                     ('version', '|S256'),
                     ]
            data = np.array([tuple([info[t[0]] for t in items])], dtype=items)
            fid.create_dataset('info', data=data, **comp_kw)
            # calibrations
            cg = fid.create_group('calibrations')
            for ci, cal in enumerate(self.info['calibrations']):
                cg.create_dataset('c%s' % ci, data=cal)

    @property
    def n_samples(self):
        """Number of time samples"""
        return len(self.times)

    def __len__(self):
        return self.n_samples

    def plot_calibration(self, title='Calibration', show=True):
        """Visualize calibration

        Parameters
        ----------
        title : str
            The title to be displayed.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object
        """
        return plot_calibration(raw=self, title=title, show=show)

    def plot(self, events=None, title='Raw', show=True):
        """Visualize calibration

        Parameters
        ----------
        events : array | None
            Events associated with the Raw instance.
        title : str
            The title to be displayed.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : matplotlib.figure.Figure instance
            The resulting figure object.
        """
        return plot_raw(raw=self, events=events, title=title, show=show)

    def plot_heatmap(self, start=None, stop=None, cmap=None, title=None,
                     vmax=None, kernel=dict(size=100, half_width=50),
                     colorbar=True, show=True):
        """ Plot heatmap of X/Y positions on canvas, e.g., screen

        Parameters
        ----------
        start : float | None
            Start time in seconds.
        stop : float | None
            End time in seconds.
        cmap : matplotlib Colormap
            The colormap to use.
        title : str
            The title to be displayed.
        vmax : float | None
            The maximum (and -minimum) value to use for the colormap.
        kernel : dict
            Parameters for the smoothing kernel (size, half_width).
        colorbar : bool
            Whether to show the colorbar.
        show : bool
            Whether to show the figure or not.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The resulting figure object
        """
        return plot_heatmap_raw(raw=self, start=start, stop=stop, cmap=cmap,
                                title=title, vmax=vmax, kernel=kernel,
                                colorbar=colorbar, show=show)

    @property
    def times(self):
        """Time values"""
        return self._times

    def time_as_index(self, times):
        """Convert time to indices

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.

        Returns
        -------
        index : ndarray
            Indices corresponding to the times supplied.
        """
        index = np.atleast_1d(times) * self.info['sfreq']
        return index.astype(int)

    def find_events(self, pattern, event_id):
        """Find parsed messages

        Parameters
        ----------
        pattern : str | callable
            A substring to be matched or a callable that matches
            a string, for example ``lambda x: 'my-message' in x``
        event_id : int
            The event id to use.

        Returns
        -------
        idx : instance of numpy.ndarray (times, event_id)
            The indices found.
        """
        return find_events(raw=self, pattern=pattern, event_id=event_id)

    def remove_blink_artifacts(self, measures=('ps'), interp='linear', borders=(0.025, 0.1),
                               use_only_blink=False):
        """Remove blink artifacts from gaze data

        This function uses the timing of saccade events to clean up
        pupil size data.

        Parameters
        ----------
        measures: list | tuple
            Measures to interpolate
        interp : str | None
            If string, can be 'linear' or 'zoh' (zeroth-order hold).
            If None, no interpolation is done, and extra ``nan`` values
            are inserted to help clean data. (The ``nan`` values inserted
            by Eyelink itself typically do not span the entire blink
            duration.)
        borders : float | list of float
            Time on each side of the saccade event to use as a border
            (in seconds). Can be a 2-element list to supply different borders
            for before and after the blink. This will be additional time
            that is eliminated as invalid and interpolated over
            (or turned into ``nan``).
        use_only_blink : bool
            If True, interpolate only over regions where a blink event
            occurred. If False, interpolate over all regions during
            which saccades occurred -- this is generally safer because
            Eyelink will not always categorize blinks correctly.
        """
        if interp is not None and interp not in ['linear', 'zoh']:
            raise ValueError('interp must be None, "linear", or "zoh", not '
                             '"%s"' % interp)
        borders = np.array(borders)
        if borders.size == 1:
            borders == np.array([borders, borders])
        blinks = self.discrete['blinks']['stime']
        starts = self.discrete['saccades']['stime']
        ends = self.discrete['saccades']['etime']
        # only use saccades that enclose a blink
        if use_only_blink:
            use = np.searchsorted(ends, blinks)
            ends = ends[use]
            starts = starts[use]
        starts = starts - borders[0]
        ends = ends + borders[1]
        # eliminate overlaps and unusable ones
        etime = (self.n_samples - 1) / self.info['sfreq']
        use = np.logical_and(starts > 0, ends < etime)
        starts = starts[use]
        ends = ends[use]
        use = starts[1:] > ends[:-1]
        starts = starts[np.concatenate([[True], use])]
        ends = ends[np.concatenate([use, [True]])]
        assert len(starts) == len(ends)
        for stime, etime in zip(starts, ends):
            sidx, eidx = self.time_as_index([stime, etime])
            ps_vals = self['ps', sidx:eidx][0]
            if len(ps_vals):
                if interp is None:
                    fix = np.nan
                elif interp == 'zoh':
                    fix = ps_vals[0]
                elif interp == 'linear':
                    len_ = eidx - sidx
                    fix = np.linspace(ps_vals[0], ps_vals[-1], len_)
                vals = self['ps', sidx:eidx][0]
                vals[:] = np.nan
                ps_vals[:] = fix

    def filter(self, l_freq, h_freq, picks=None, filter_length='auto',
               l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1,
               method='fir', iir_params=None, phase='zero',
               fir_window='hamming', fir_design='firwin',
               pad='reflect_limited'):
        """Filter a subset of channels.
        Applies a zero-phase low-pass, high-pass, band-pass, or band-stop
        filter to the channels selected by ``picks``. By default the data
        of the Raw object is modified inplace.
        The Raw object has to have the data loaded e.g. with ``preload=True``
        or ``self.load_data()``.
        ``l_freq`` and ``h_freq`` are the frequencies below which and above
        which, respectively, to filter out of the data. Thus the uses are:
            * ``l_freq < h_freq``: band-pass filter
            * ``l_freq > h_freq``: band-stop filter
            * ``l_freq is not None and h_freq is None``: high-pass filter
            * ``l_freq is None and h_freq is not None``: low-pass filter
        ``self.info['lowpass']`` and ``self.info['highpass']`` are only
        updated with picks=None.
        .. note:: If n_jobs > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporaily stored in memory.
        Parameters
        ----------
        l_freq : float | None
            Low cut-off frequency in Hz. If None the data are only low-passed.
        h_freq : float | None
            High cut-off frequency in Hz. If None the data are only
            high-passed.
        picks : array-like of int | None
            Indices of channels to filter. If None only the data (MEG/EEG)
            channels will be filtered.
        filter_length : str | int
            Length of the FIR filter to use (if applicable):
            * 'auto' (default): the filter length is chosen based
              on the size of the transition regions (6.6 times the reciprocal
              of the shortest transition band for fir_window='hamming'
              and fir_design="firwin2", and half that for "firwin").
            * str: a human-readable time in
              units of "s" or "ms" (e.g., "10s" or "5500ms") will be
              converted to that number of samples if ``phase="zero"``, or
              the shortest power-of-two length at least that duration for
              ``phase="zero-double"``.
            * int: specified length in samples. For fir_design="firwin",
              this should not be used.
        l_trans_bandwidth : float | str
            Width of the transition band at the low cut-off frequency in Hz
            (high pass or cutoff 1 in bandpass). Can be "auto"
            (default) to use a multiple of ``l_freq``::
                min(max(l_freq * 0.25, 2), l_freq)
            Only used for ``method='fir'``.
        h_trans_bandwidth : float | str
            Width of the transition band at the high cut-off frequency in Hz
            (low pass or cutoff 2 in bandpass). Can be "auto"
            (default) to use a multiple of ``h_freq``::
                min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)
            Only used for ``method='fir'``.
        n_jobs : int | str
            Number of jobs to run in parallel. Can be 'cuda' if scikits.cuda
            is installed properly, CUDA is initialized, and method='fir'.
        method : str
            'fir' will use overlap-add FIR filtering, 'iir' will use IIR
            forward-backward filtering (via filtfilt).
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering.
            See mne.filter.construct_iir_filter for details. If iir_params
            is None and method="iir", 4th order Butterworth will be used.
        phase : str
            Phase of the filter, only used if ``method='fir'``.
            By default, a symmetric linear-phase FIR filter is constructed.
            If ``phase='zero'`` (default), the delay of this filter
            is compensated for. If ``phase=='zero-double'``, then this filter
            is applied twice, once forward, and once backward. If 'minimum',
            then a minimum-phase, causal filter will be used.
            .. versionadded:: 0.13
        fir_window : str
            The window to use in FIR design, can be "hamming" (default),
            "hann" (default in 0.13), or "blackman".
            .. versionadded:: 0.13
        fir_design : str
            Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
            or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
            a time-domain design technique that generally gives improved
            attenuation using fewer samples than "firwin2".
            .. versionadded:: 0.15
        skip_by_annotation : str | list of str
            If a string (or list of str), any annotation segment that begins
            with the given string will not be included in filtering, and
            segments on either side of the given excluded annotated segment
            will be filtered separately (i.e., as independent signals).
            The default (``('edge', 'bad_acq_skip')`` will separately filter
            any segments that were concatenated by :func:`mne.concatenate_raws`
            or :meth:`mne.io.Raw.append`, or separated during acquisition.
            To disable, provide an empty list.
            .. versionadded:: 0.16.
        pad : str
            The type of padding to use. Supports all :func:`numpy.pad` ``mode``
            options. Can also be "reflect_limited" (default), which pads with a
            reflected version of each vector mirrored on the first and last
            values of the vector, followed by zeros.
            Only used for ``method='fir'``.
            .. versionadded:: 0.15
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        Returns
        -------
        raw : instance of Raw
            The raw instance with filtered data.
        See Also
        --------
        mne.Epochs.savgol_filter
        mne.io.Raw.notch_filter
        mne.io.Raw.resample
        mne.filter.filter_data
        mne.filter.construct_iir_filter
        Notes
        -----
        For more information, see the tutorials
        :ref:`sphx_glr_auto_tutorials_plot_background_filtering.py`
        and
        :ref:`sphx_glr_auto_tutorials_plot_artifacts_correction_filtering.py`.
        """

        filter_data(
            self._samples, self.info['sfreq'], l_freq, h_freq,
            picks, filter_length, l_trans_bandwidth, h_trans_bandwidth,
            n_jobs, method, iir_params, copy=False, phase=phase,
            fir_window=fir_window, fir_design=fir_design, pad=pad)
        # update info if filter is applied to all data channels,
        # and it's not a band-stop filter
        _filt_update_info(self.info, True, l_freq, h_freq)

        return self



def read_raw(fname):
    """General Eye-tracker Reader

    Parameters
    ----------
    fname : str
        The name of the eye-tracker data file.
        Files currently supported are EDF and HD5
    """
    _, ext = op.splitext(fname)
    if ext == '.edf':
        from .edf._raw import RawEDF
        raw = RawEDF(fname)
    elif ext == '.hd5':
        from .hd5._raw import RawHD5
        raw = RawHD5(fname)
    elif ext == '.hdf5':
        from .hd5._raw import RawIoHubHD5
        raw = RawIoHubHD5(fname)
    return raw
