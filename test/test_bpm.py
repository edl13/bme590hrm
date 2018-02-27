import pytest
import logging
from logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_import_data():
    log.debug('Begin bpm detection testing')
    from HeartRateMonitor import HeartRateMonitor
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv')
    hrm.detect_bpm()
    assert hrm.mean_hr_bpm == pytest.approx(75, 10)
    log.debug('End bpm detection testing')
