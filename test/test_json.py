import pytest
import logging
import json
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_json():
    log.debug('Begin data output testing')
    from ..HeartRateMonitor import HeartRateMonitor

    hrm = HeartRateMonitor(
        filename='test_data/test_data1.csv', t_units='s', v_units='V')
    hrm.detect_bpm()
    hrm.detect_voltage_extremes()
    hrm.get_duration()
    hrm.get_peaks()

    hrm.export_json()
    data_dict = json.load(open('test_data/test_data1.json'))

    beat_times_compare = [243.0, 1122.0, 2001.0,
                          2853.0, 3705.0, 4557.0, 5439.0, 5805.0, 6147.0,
                          7221.0, 8130.0, 9006.0, 9861.0, 10692.0, 11601.0,
                          12522.0, 13410.0, 14307.0, 15192.0, 16053.0, 16911.0,
                          17769.0, 18654.0, 19593.0, 20484.0, 21330.0, 22185.0,
                          23022.0, 23871.0, 24750.0, 25629.0, 26523.0, 27438.0,
                          28305.0, 29142.0]

    data_compare = {
        'BPM': 70.9219858,
        'Voltage Min': -680,
        'Voltage Max': 1050,
        'Duration': 27775,
        'Number of Beats': 35,
        'Beat Times': beat_times_compare}

    for key in data_dict:
        assert data_dict[key] == pytest.approx(data_compare[key])

    log.debug('End data output testing')
