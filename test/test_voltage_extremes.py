import pytest
import logging
from ..logging_config import config
# from ...logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_import_data():
    log.debug('Begin voltage extreme detection testing')
    from ..HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor('test_data/test_data1.csv', t_units='s',
                           v_units='V')
