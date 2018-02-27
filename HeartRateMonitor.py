'''
Heart rate monitor class which characterizes ECG signal using autocorrelation
and other processing techniques
'''

import numpy as np
import pandas as pd
from scipy import signal


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

    def detect_bpm(self):
        data = self.data
        t = data[:, 0]
        v = data[:, 1]
        dt = t[1] - t[0]

        # Remove dc offsets
        avg_len = 200
        v_dc = v - np.convolve(v, np.ones(avg_len) / avg_len, mode='same')

        # Autocorrelation
        std = np.std(v_dc)
        corr = np.correlate(v_dc, v_dc, mode='full')
        corr = corr[int(len(corr) / 2):]
        corr = np.divide(corr, pow(std, 2))
        # Autocorrelation peak detection with scipy.
        peaks = signal.find_peaks_cwt(corr, np.arange(1, 20), noise_perc=50,
                                      min_snr=1.5)

        # Record peaks with correlation > 20
        corr_thresh = 20
        lags = peaks[corr[peaks] > corr_thresh]

        # Calculate BPM
        period = lags[1] - lags[0]
        bpm = 60 / (dt * period)
        print(bpm)
        self.mean_hr_bpm = bpm
        return corr, peaks, lags
