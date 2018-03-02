'''
Heart rate monitor class which characterizes ECG signal using autocorrelation
and other processing techniques
'''

import numpy as np
import pandas as pd
from scipy import signal
import logging
import matplotlib as mpl
import os
import warnings
import json
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(filename='out.log', level=logging.DEBUG)
log = logging.getLogger(__name__)


class HeartRateMonitor(object):
    '''Main heart rate monitor class to perform various characterizations of
    ECG signal
    '''

    def __init__(self, data=None, filename=None, t_units='ms', v_units='mV'):
        '''Initialize HeartRateMonitor object

        :param data: 2D numpy array with time values in the first column and
        ECG voltage values in the second column. Defaults to None.
        :param filename: CSV file with time in first column and voltage in the
        second column. Defaults to None.
        :param t_units: Time units, either 'ms', 's', or 'min'. Defaults to
        's'.
        :param v_units: Voltage units, either 'mV' or 'V'
        '''

        log.info('Initalize HeartRateMonitor')

        self.t_units = t_units
        self.v_units = v_units
        self.__t_converter = None
        self.__v_converter = None
        (self.__t_converter, self.__v_converter) = self.__get_converters(
            self.t_units, self.v_units)
        log.debug('''T units/conversion {}/{}. V units/converstion
                  {}/{}'''.format(self.t_units, self.__t_converter,
                                  self.v_units, self.__v_converter))

        if data is None and filename is None:
            self.data = []
        elif data is not None:
            self.data = data
        elif filename is not None:
            self.filename = filename
            self.import_data(filename)
        else:
            self.data = []

        self.__clean_data()

        log.debug('Converting data to ms and mV')
        self.__convert_data()

        self.mean_hr_bpm = None
        self.voltage_extremes = None
        self.duration = None
        self.num_beats = None
        self.beats = None
        self.__filt_data = None

        log.debug('Filtering data')
        self.__filter_data()

    @property
    def filename(self):
        '''Filename of imported data'''
        return self.__filename

    @filename.setter
    def filename(self, filename):
        '''Setter for filename

        :param filename: Filename'''

        self.__filename = filename

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
        '''Import data from file

        :param filename: csv file to import from
        '''

        df = pd.read_csv(filename, names=['Time', 'Voltage'])
        data = df.as_matrix()
        self.data = data
        log.info('Successfully imported {}'.format(filename))

    def __convert_data(self):
        self.data[:, 0] *= self.__t_converter
        self.data[:, 1] *= self.__v_converter

    def detect_bpm(self, time=None, units=None):
        '''Detects BPM using autocorrelation.

        :param time: Time over which to find mean BPM. Defaults to find mean
            from beginning to end of given signal. If scalar given, mean is
            found from t = 0 to t = time seconds. If two element list or
            tuple of times is given, mean is found between the two times.
            Begin and end sample points chosen to be as close to given
            arguments as possible.
        :param units: Time units of the time limits parameter
        :returns: Beats per minute
        :raise IndexError: Only one beat detected in time limits, cannot find
            BPM
        '''

        if units is None:
            units = self.t_units

        data = self.data
        t_lim = None
        (lim_converter, v_con_temp) = self.__get_converters(
            units, self.v_units)
        t_raw = data[:, 0]
        dt = t_raw[1] - t_raw[0]

        log.info('dt found to be {}'.format(dt))

        if time is None:
            t_lim = np.array((0, max(t_raw)))
        elif isinstance(time, (list, tuple)):
            if (len(time) == 2):
                time = np.array(time)
                time *= lim_converter
                t_lim = time
            else:
                raise ValueError('''Iterable time input must have two elements
                      for start and end times''')
                log.error(
                    '''Iterable time input must have two elements for start
                          and end times''')
        elif isinstance(time, (int, float)):
            time *= lim_converter
            t_lim = (0, time)
        else:
            raise TypeError('''Time argument takes scalar or two element
                  iterable''')
            log.error('Time argument takes scalar or two element iterable.')

        (start, end) = self.find_nearest_limits(t_raw, t_lim)
        log.info('''Closest start time: {}. Closest end time:
                 {}'''.format(t_raw[start], t_raw[end]))

        v = self.__filt_data[start:end]

        # Remove dc offsets
        corr1 = np.correlate(v, v, mode='full')
        corr1 = np.divide(corr1, max(corr1))
        corr1 = corr1[int(len(corr1) / 2) + 000:]

        # Autocorrelation peak detection with scipy.
        widths = np.arange(1, 400)
        peaks = signal.find_peaks_cwt(
            corr1,
            widths,
            noise_perc=10,
            min_snr=20,
            max_distances=np.divide(widths, 10))

        # Calculate BPM
        try:
            period = peaks[1] - peaks[0]
        except IndexError:
            log.error('''Only one peak detected in time region specified.
                      Expand time region to detect BPM.''')
            raise IndexError(
                '''Only one peak detected in time region specified.
                              Unable to detect BPM''')

        bpm = 60 * self.__t_converter / (dt * period)
        self.mean_hr_bpm = bpm
        plt.plot(corr1)
        plt.plot(peaks, np.zeros(len(peaks)), 'o')
        plt.ion()
        plt.show()
        log.info('BPM found to be {}'.format(bpm))
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

    def detect_voltage_extremes(self, thresh=None, units=None):
        '''Detect voltage extremes above positive and negative threshold.
            Returns maximum and minimum voltages.

        :param thresh: Positive threshold voltage for extreme values (Defaults
            to +- 300mV)
        :param units: Units of threshold. Defaults to class units
        :return: Tuple (minimum voltage, maximum voltage)
        '''

        if units is None:
            units = self.v_units

        (t_converter, v_converter) = self.__get_converters(self.t_units, units)

        if thresh is None:
            thresh = 300 / v_converter

        thresh_conv = thresh * v_converter

        t_thresh = np.where(np.abs(self.data[:, 1]) >= thresh_conv)[0]
        log.debug('V thresh set to {} mV'.format(thresh_conv))
        log.debug('{} data points outside thresh'.format(len(t_thresh)))

        if len(t_thresh) > 0:
            for t in t_thresh:
                warnings.warn('''Extreme voltage above {}{} of {}{} found at
                              {}{}'''.format(
                    thresh, units,
                    np.divide(self.data[t, 1],
                              self.__v_converter), self.v_units,
                    np.divide(t, self.__t_converter), self.t_units))

        max_v = np.max(self.data[:, 1])
        min_v = np.min(self.data[:, 1])

        log.info('(min, max) voltage set to {}'.format((min_v, max_v)))

        self.voltage_extremes = (min_v, max_v)
        return (min_v, max_v)

    def __get_converters(self, t_units, v_units):
        if type(t_units) is not str:
            log.error('Non-string time units')
            raise TypeError('Please input string for time units')

        if type(v_units) is not str:
            log.error('Non-string voltage units')
            raise TypeError('Please input string for voltage units')

        if (t_units == 's'):
            t_converter = 1000
        elif (t_units == 'ms'):
            t_converter = 1
        elif (t_units == 'min'):
            t_converter = 60000
        else:
            log.error('Unknown time units of {}'.format(t_units))
            raise ValueError('Time units must be \'s\', \'ms\', or \'min\'.')

        if (v_units == 'V'):
            v_converter = 1000
        elif (v_units == 'mV'):
            v_converter = 1
        else:
            log.error('Unknown voltage units of {}'.format(v_units))
            raise ValueError('Voltage units must be \'mV\' or \'V\'.')

        return (t_converter, v_converter)

    def __filter_data(self):
        '''Filter raw data with 5-15 Hz passband according to Pan-Tompkins
        algorithm, then rectified and squared'''

        dt = self.data[1, 0] - self.data[0, 0]  # dt in ms
        nyq = (1 / (dt / 1000)) * 0.5
        log.info('Nyquist frequency found to be {} Hz'.format(nyq))

        low = 5 / nyq
        hi = 15 / nyq
        log.info('Cutoff frequencies set to {} to {} Hz'.format(low, hi))

        b, a = signal.butter(2, (low, hi), btype='bandpass')
        filt = signal.lfilter(b, a, self.data[:, 1])

        # Rectify
        filt[filt < 0] = 0

        # Square
        filt = np.multiply(filt, filt)

        self.__filt_data = filt

    def get_peaks(self):
        '''Detect peaks and return timing of beats array

        :return beats: Beat times array in ms'''

        widths = np.arange(1, 400)

        log.info('Begin peak detection')
        peaks = signal.find_peaks_cwt(
            self.__filt_data,
            widths,
            noise_perc=10,
            min_snr=20,
            max_distances=np.divide(widths, 10))

        dt = self.data[1, 0] - self.data[0, 0]

        self.beats = np.multiply(peaks, dt)
        self.num_beats = len(peaks)
        log.info('{} beats found in signal'.format(len(peaks)))

        return (peaks)

    def get_duration(self):
        '''Find signal duration

        :return duration: Total duration'''

        dur = max(self.data[:, 0]) - min(self.data[:, 0])
        log.info('Duration of ECG found to be {} ms'.format(dur))
        self.duration = dur
        return dur

    def __clean_data(self):
        '''Find NaN in input data and fixes gap'''

        log.debug('Begin cleaning data')
        interp_t = 0
        interp_v = 0
        for i, t in enumerate(self.data[:, 0]):
            if np.isnan(t):
                if (i == 0):
                    interp_t = self.data[i + 1, 0]
                elif i == len(self.data[:, 0]) - 1:
                    interp_t = self.data[i - 1, 0]
                else:
                    interp_t = (self.data[i - 1, 0] + self.data[i + 1, 0]) / 2

                warnings.warn('''Blank time value at index {} interpolating as
                              {}'''.format(i, interp_t))

                log.info('''Blank time value at index {} interpolating as
                              {}'''.format(i, interp_t))
                self.data[i, 0] = interp_t

        for i, t in enumerate(self.data[:, 1]):
            if np.isnan(t):
                log.debug('{}{}'.format(t, np.isnan(t)))
                if (i == 0):
                    interp_v = self.data[i + 1, 1]
                elif i == len(self.data[:, 1]) - 1:
                    interp_v = self.data[i - 1, 1]
                else:
                    interp_v = (self.data[i - 1, 1] + self.data[i + 1, 1]) / 2

                warnings.warn('''Blank voltage value at index {} interpolating
                              as {}'''.format(i, interp_v))

                log.info('''Blank voltage value at index {} interpolating
                              as {}'''.format(i, interp_v))
                self.data[i, 1] = interp_v

    def export_json(self, filename=None):
        '''Export ECG characteristics as JSON file

        :param filename: Filename to store as. Default is input filename as
            .json
        '''

        data_dict = {
            'BPM': self.mean_hr_bpm,
            'Voltage Min': self.voltage_extremes[0],
            'Voltage Max': self.voltage_extremes[1],
            'Duration': self.duration,
            'Number of Beats': self.num_beats,
            'Beat Times': self.beats.tolist()
        }

        if filename is None:
            if self.filename is not None:
                csv_name = self.filename
                filename = os.path.splitext(csv_name)[0] + '.json'
                log.info('Filename is {}'.format(filename))
            else:
                raise ValueError('''No filename specified at object
                                 initialization or at export_json call''')

        log.info('Writing json to {}'.format(filename))

        with open(filename, 'w') as output:
            json.dump(data_dict, output)
