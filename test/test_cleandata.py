import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_cleandata():
    log.debug('Begin data cleaning testing')
    from ..HeartRateMonitor import HeartRateMonitor
    with pytest.warns(UserWarning):
        hrm = HeartRateMonitor(filename='test_data/test_data28.csv',
                               t_units='s', v_units='V')

    assert hrm.data[324, 0] == pytest.approx(900)
    assert hrm.data[338, 1] == pytest.approx(-345)

    log.debug('End data cleaning testing')
