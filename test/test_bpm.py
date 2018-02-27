import pytest
import logging
from logging_config import config

# init logging config
logging.basicConfig(**config)
log = logging.getLogger(__name__)


def test_import_data():
    log.debug('Begin bpm detection testing')
    from HeartRateMonitor import HeartRateMonitor
    filenames = ['test_data1.csv', 'test_data10.csv', 'test_data11.csv']
    # Approximate BPM by manual calculations
    bpms = [75, 95, 66]
    for i, name in enumerate(filenames):
        hrm = HeartRateMonitor(filename='test_data/' + name)
        hrm.detect_bpm()
        assert hrm.mean_hr_bpm == pytest.approx(bpms[i], 10)
    log.debug('End bpm detection testing')
