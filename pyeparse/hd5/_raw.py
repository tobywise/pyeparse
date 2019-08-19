# -*- coding: utf-8 -*-
"""HD5 Raw class"""

import numpy as np
from os import path as op
from copy import deepcopy
from datetime import datetime

from .._baseraw import _BaseRaw


class RawHD5(_BaseRaw):
    """Represent HD5 files in Python

    Parameters
    ----------
    fname : str
        The name of the EDF file.
    """
    def __init__(self, fname):
        try:
            import h5py
        except:
            raise ImportError('h5py is required but was not found')
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        info = dict()
        with h5py.File(fname, mode='r') as fid:
            # samples
            samples = np.array(fid['samples'])
            info['sample_fields'] = list(deepcopy(samples.dtype.names))
            samples = samples.view(np.float64).reshape(samples.shape[0], -1).T
            # times
            times = np.array(fid['times'])
            # discrete
            discrete = dict()
            dg = fid['discrete']
            for key in dg.keys():
                discrete[key] = np.array(dg[key])
            # info
            data = np.array(fid['info'])
            for key in data.dtype.names:
                info[key] = data[key][0]
            date = info['meas_date'].decode('ASCII')
            info['meas_date'] = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            # calibrations
            cg = fid['calibrations']
            cals = np.array([np.array(cg['c%s' % ii])  # maintain order
                             for ii in range(len(cg.keys()))])
            info['calibrations'] = cals

        self._samples = samples
        self._times = times
        self.discrete = discrete
        self.info = info
        _BaseRaw.__init__(self)  # perform sanity checks



class RawIoHubHD5(_BaseRaw):
    """Represent HD5 files in Python

    Parameters
    ----------
    fname : str
        The name of the EDF file.
    """
    def __init__(self, fname, fields=('gaze_x', 'gaze_y', 'pupil_measure1'), screen_size=(1024,768),
                 field_names=('xpos', 'ypos', 'ps')):
        try:
            import h5py
        except:
            raise ImportError('h5py is required but was not found')
        if not op.isfile(fname):
            raise IOError('file "%s" not found' % fname)
        info = dict()
        with h5py.File(fname, mode='r') as fid:

            gaze = fid['data_collection']['events']['eyetracker']['MonocularEyeSampleEvent'][...]
            samples = gaze[list(fields)]
            info['sample_fields'] = list(field_names)
            samples = samples.copy().view(np.float32).reshape(samples.shape[0], -1).T
            samples = samples.astype(np.float64)
            samples[0, :] += screen_size[0] / 2
            samples[1, :] += screen_size[1] / 2

            # times
            times = np.array(fid['data_collection']['events']['eyetracker']['MonocularEyeSampleEvent']['time'])
            start_time = times[0]
            times -= start_time

            # reorder things
            info['sfreq'] = float(np.round(times[times < 1].shape[0] / 10, -1) * 10)
            times_sorted_idx = times.argsort()
            times = times[times_sorted_idx]

            orig_times = times  # original times
            # reorder incorrectly ordered samples
            samples = samples[:, times_sorted_idx]
            times = np.arange(len(orig_times), dtype=np.float64) / info['sfreq']

            # discrete
            discrete = dict()

            # blinks
            blink_start = fid['data_collection']['events']['eyetracker']['BlinkEndEvent']['time'] - \
                        fid['data_collection']['events']['eyetracker']['BlinkEndEvent']['duration']
            blink_end = fid['data_collection']['events']['eyetracker']['BlinkEndEvent']['time']
            dtype = [('stime', np.float64), ('etime', np.float64)]
            n_blinks = len(blink_start)
            discrete['blinks'] = np.empty(n_blinks, dtype=dtype)
            discrete['blinks']['stime'] = np.array(blink_start - start_time, dtype=np.float64)
            discrete['blinks']['etime'] = np.array(blink_end - start_time, dtype=np.float64)

            # saccades
            saccade_start = fid['data_collection']['events']['eyetracker']['SaccadeEndEvent']['time'] - \
                        fid['data_collection']['events']['eyetracker']['SaccadeEndEvent']['duration']
            saccade_end = fid['data_collection']['events']['eyetracker']['SaccadeEndEvent']['time']
            n_saccades = len(saccade_start)
            discrete['saccades'] = np.empty(n_saccades, dtype=dtype)

            discrete['saccades']['stime'] = np.array(saccade_start - start_time, dtype=np.float64)
            discrete['saccades']['etime'] = np.array(saccade_end - start_time, dtype=np.float64)

            # fixations
            fix_start = fid['data_collection']['events']['eyetracker']['FixationEndEvent']['time'] - \
                        fid['data_collection']['events']['eyetracker']['FixationEndEvent']['duration']
            fix_end = fid['data_collection']['events']['eyetracker']['FixationEndEvent']['time']
            xpos = fid['data_collection']['events']['eyetracker']['FixationEndEvent']['average_gaze_x']
            ypos = fid['data_collection']['events']['eyetracker']['FixationEndEvent']['average_gaze_y']

            dtype = [('stime', np.float64), ('etime', np.float64), ('xpos', np.float64), ('ypos', np.float64)]
            n_fixations = len(fix_start)
            discrete['fixations'] = np.empty(n_fixations, dtype=dtype)
            discrete['fixations']['stime'] = np.array(fix_start - start_time, dtype=np.float64)
            discrete['fixations']['etime'] = np.array(fix_end - start_time, dtype=np.float64)
            discrete['fixations']['xpos'] = np.array(xpos, dtype=np.float64)
            discrete['fixations']['ypos'] = np.array(ypos, dtype=np.float64)


            # messages
            off = fid['data_collection']['events']['experiment']['MessageEvent']['msg_offset']
            dtype = [('stime', np.float64), ('msg', '|S%s' % 100)]
            discrete['messages'] = np.empty((len(off)), dtype=dtype)
            discrete['messages']['stime'] = fid['data_collection']['events']['experiment']['MessageEvent']['time'] - start_time
            discrete['messages']['msg'] = fid['data_collection']['events']['experiment']['MessageEvent']['text']

            for key in discrete:
                for sub_key in ('stime', 'etime'):
                    if sub_key in discrete[key].dtype.names:
                        _adjust_time(discrete[key][sub_key], orig_times, times)
            # info
            info['subject'] = fid['data_collection']['session_meta_data'][...]['code'][0]

            # filtering
            info['lowpass'] = None
            info['highpass'] = None

            # approximate sampling frequency
            info['screen_coords'] = screen_size
            info['ps_units'] = 'diameter'

        self._samples = samples
        self._times = times
        self.info = info
        self.discrete = discrete


def _adjust_time(x, orig_times, times):
    """Helper to adjust time, inplace"""
    x[:] = np.interp(x, orig_times, times)


def _remove_extra_samples(start, end, other=None):

    # start = fid['data_collection']['events']['eyetracker']['FixationStartEvent']['time'].copy()
    # end = fid['data_collection']['events']['eyetracker']['FixationEndEvent']['time'].copy()
    #
    # other = {'end': [fid['data_collection']['events'][
    #                      'eyetracker']['FixationEndEvent'][
    #                      'average_gaze_x'],
    #                  fid['data_collection']['events'][
    #                      'eyetracker']['FixationEndEvent'][
    #                      'average_gaze_y']]}

    if other is None:
        other = dict()

    n_blinks = np.max([len(start), len(end)])
    start_pad = np.round(np.pad(start, (0, n_blinks + 1 - len(start)), 'constant', constant_values=np.nan), 3)
    end_pad = np.round(np.pad(end, (0, n_blinks + 1 - len(end)), 'constant', constant_values=np.nan), 3)
    missing_end = np.roll(end_pad, 1) - start_pad > 0
    missing_start = start_pad > end_pad

    while len(start) != len(end) or np.any(missing_end) or np.any(missing_start):
        # print len(start)
        # print len(end)

        if np.any(missing_end):
            idx = np.where(missing_end)[0][0] - 1
            start = np.delete(start, idx)
            if 'start' in other:
                for i in range(len(other['start'])):
                    other['start'][i] = np.delete(other['start'][i], idx)

        elif np.any(missing_start):
            idx = np.where(missing_start)[0][0]
            end = np.delete(end, idx)
            if 'end' in other:
                for i in range(len(other['end'])):
                    other['end'][i] = np.delete(other['end'][i], idx)

        elif len(start) - len(end) == 1:  # end of final blink wasn't captured
            start = start[:-1]
            if 'start' in other:
                for i in range(len(other['start'])):
                    other['start'][i] = other['start'][i][:-1]

        else:  # if we just have random samples at the end for no apparent reason
            n = np.min([len(start), len(end)])
            start = start[:n]
            end = end[:n]

            if 'end' in other:
                for i in range(len(other['end'])):
                    other['end'][i] = other['end'][i][:n]
            if 'start' in other:
                for i in range(len(other['start'])):
                    other['start'][i] = other['start'][i][:n]

        n_blinks = np.max([len(start), len(end)])
        start_pad = np.round(np.pad(start, (0, n_blinks +1 - len(start)), 'constant', constant_values=np.nan), 3)
        end_pad = np.round(np.pad(end, (0, n_blinks + 1- len(end)), 'constant', constant_values=np.nan), 3)
        missing_end = np.roll(end_pad, 1) - start_pad > 0
        missing_start = start_pad > end_pad

    return start, end, other
