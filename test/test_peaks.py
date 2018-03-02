import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_peaks():
    log.debug('Begin units testing')
    from ..HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv', t_units='s',
                           v_units='V')
    hrm.get_peaks()

    assert hrm.num_beats == pytest.approx(35, 0.2)

    log.debug('End units testing')
