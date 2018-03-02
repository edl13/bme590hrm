import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_duration():
    log.debug('Begin duration testing')
    from ..HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv', t_units='s',
                           v_units='V')
    hrm.get_duration()

    assert hrm.duration == pytest.approx(27775)

    log.debug('End units testing')
