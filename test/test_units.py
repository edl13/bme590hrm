import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_import_data():
    log.debug('Begin units testing')
    from ..HeartRateMonitor import HeartRateMonitor
    with pytest.raises(ValueError):
        hrm = HeartRateMonitor(filename='test_data/test_data1.csv',
                               t_units='hr')
    with pytest.raises(ValueError):
        hrm = HeartRateMonitor(filename='test_data/test_data1.csv',
                               v_units='JC^-1')

    hrm = HeartRateMonitor(filename='test_data/test_data1.csv', t_units='s',
                           v_units='V')

    assert hrm._HeartRateMonitor__t_converter == 1000
    assert hrm._HeartRateMonitor__v_converter == 1000
    assert hrm.data[1, 0] == 3
    log.debug('End units testing')
