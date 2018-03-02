import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_voltage_extremes():
    log.debug('Begin voltage extreme detection testing')
    from ..HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv', t_units='s',
                           v_units='V')
    with pytest.warns(UserWarning):
        hrm.detect_voltage_extremes(thresh=300, units='mV')
    assert hrm.voltage_extremes == pytest.approx((-680, 1050))

    hrm = HeartRateMonitor(filename='test_data/test_data32.csv', t_units='ms',
                           v_units='mV')
    with pytest.warns(UserWarning):
        hrm.detect_voltage_extremes(thresh=300, units='mV')
