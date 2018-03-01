'''
Heart rate monitor class which characterizes ECG signal using autocorrelation
and other processing techniques
'''

import numpy as np
import pandas as pd
from scipy import signal
import logging
from . logging_config import config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


class HeartRateMonitor(object):
    '''Main heart rate monitor class to perform various characterizations of
    ECG signal
    '''

    def __init__(self, data=None, filename=None):
        '''Initialize HeartRateMonitor object

        :param data: 2D numpy array with time values in the first column and
        ECG voltage values in the second column. Defaults to None.
        :param filename: CSV file with time in first column and voltage in the
        second column. Defaults to None.
        '''
        if data is None and filename is None:
            self.data = []
        elif data is not None:
            self.data = data
        elif filename is not None:
            self.import_data(filename)
        else:
            self.data = []

        self.mean_hr_bpm = None
        self.voltage_extremes = None
        self.duration = None
        self.num_beats = None
        self.beats = None

    @property
    def data(self):
        '''Internal time-dependent ECG data property'''
        return self.__data

    @data.setter
    def data(self, data):
        '''Set data
        :param data: ECG values to set
        '''
        self.__data = data

    @property
    def mean_hr_bpm(self):
        '''Mean bpm over specified amount of time'''
        return self.__mean_hr_bpm

    @mean_hr_bpm.setter
    def mean_hr_bpm(self, bpm):
        '''Set mean_hr_bpm
        :param bpm: Mean bpm
        '''
        self.__mean_hr_bpm = bpm

    @property
    def voltage_extremes(self):
        '''Minimum and maximum lead voltages'''
        return self.__voltage_extremes

    @voltage_extremes.setter
    def voltage_extremes(self, voltages):
        '''Set voltage_extremes
        :param voltages: Tuple of min and max voltages
        '''
        self.__voltage_extremes = voltages

    @property
    def duration(self):
        '''Duration of ECG strip'''
        return self.__duration

    @duration.setter
    def duration(self, duration):
        '''Set duration
        :param duration: Duration of ECG
        '''
        self.__duration = duration

    @property
    def num_beats(self):
        '''Number of beats detected'''
        return self.__num_beats

    @num_beats.setter
    def num_beats(self, num_beats):
        '''Set num_beats
        :param num_beats: Number of beats detected
        '''
        self.__num_beats = num_beats

    @property
    def beats(self):
        '''Numpy array of times beats occured'''
        return self.__beats

    @beats.setter
    def beats(self, beats):
        '''Set beats
        :param beats: Numpy array of beat times
        '''
        self.__beats = beats

    def import_data(self, filename):
        df = pd.read_csv(filename, names=['Time', 'Voltage'])
        data = df.as_matrix()
        self.data = data

    def detect_bpm(self, time=None):
        '''Detects BPM using autocorrelation.
        :param time: Time over which to find mean BPM. Defaults to find mean
        from beginning to end of given signal. If scalar given, mean is found
        from t = 0 to t = time seconds. If two element list or tuple of times
        is given, mean is found between the two times. Begin and end sample
        points chosen to be as close to given arguments as possible.'''

        data = self.data
        t = data[:, 0]
        v = data[:, 1]
        dt = t[1] - t[0]
        t_lim = None

        if time is None:
            t_lim = (0, max(t))
        elif isinstance(time, (list, tuple)):
            if(len(time) == 2):
                t_lim = time
            else:
                raise ValueError('''Iterable time input must have two elements
                      for start and end times''')
                log.error('''Iterable time input must have two elements for start
                          and end times''')
        elif isinstance(time, (int, float)):
            t_lim = (0, time)
        else:
            raise TypeError('''Time argument takes scalar or two element
                  iterable''')
            log.error('Time argument takes scalar or two element iterable.')

        (start, end) = self.find_nearest_limits(t, t_lim)
        # Remove dc offsets
        avg_len = 50
        v_dc = v - np.convolve(v, np.ones(avg_len) / avg_len, mode='same')

        # Autocorrelation
        # std = np.std(v_dc)
        corr1 = np.correlate(v_dc, v_dc, mode='full')
        corr1 = np.divide(corr1, max(corr1))
        # corr2 = np.correlate(corr1, corr1, mode='full')
        corr1 = corr1[int(len(corr1) / 2):]

        # Autocorrelation peak detection with scipy.
        widths = np.arange(1, 20)
        peaks = signal.find_peaks_cwt(corr1, widths, noise_perc=70,
                                      min_snr=45,
                                      max_distances=np.divide(widths, 2))

        # Record peaks with positive correlation
        corr_thresh = 0
        lags = peaks[corr1[peaks] > corr_thresh]

        # Calculate BPM
        period = lags[1] - lags[0]
        bpm = 60 / (dt * period)
        self.mean_hr_bpm = bpm
        return bpm

    def find_nearest_limits(self, t, t_lim):
        '''Find nearest t values to given limits

        :param t: Array of sample times
        :param t_lim: Two element tuple of start and end times
        :return: Tuple of start and end indices of t
        '''

        begin = t_lim[0]
        end = t_lim[1]

        begin_i = np.argmin(np.abs(t - begin))
        end_i = np.argmin(np.abs(t - end))
        return (begin_i, end_i)

    def detect_voltage_extremes(self, thresh=300):
        pass
