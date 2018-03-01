import numpy as np
import pytest
import logging
from ..logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_import_data():
    log.debug('Begin data import testing')
    from ..HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv')
    data = hrm.data
    np.testing.assert_almost_equal(data[4], [0.011, -0.145], decimal=5)
    with pytest.raises(OSError):
        HeartRateMonitor(filename='test_data/wrong.csv')
    log.debug('End data import testing')
