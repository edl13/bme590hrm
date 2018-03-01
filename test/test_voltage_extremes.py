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
    filenames = ['test_data1.csv', 'test_data10.csv', 'test_data11.csv']
    # Approximate BPM by manual calculations
    bpms = [75, 95, 66]
    for i, name in enumerate(filenames):
        hrm = HeartRateMonitor(filename='test_data/' + name)
        hrm.detect_bpm()
        assert hrm.mean_hr_bpm == pytest.approx(bpms[i], 10)
    log.debug('End bpm detection testing')

    for i, name in enumerate(filenames):
        hrm = HeartRateMonitor(filename='test_data/' + name)
        hrm.detect_bpm(time=(5, 10))
        assert hrm.mean_hr_bpm == pytest.approx(bpms[i], 10)

    # Test time input
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv')
    with pytest.raises(ValueError):
        hrm.detect_bpm(time=(1, 2, 3))
    with pytest.raises(TypeError):
        hrm.detect_bpm(time='Hello')

    # Test finding limits
    hrm = HeartRateMonitor(filename='test_data/test_data1.csv')
    lim = hrm.find_nearest_limits(hrm.data[:, 0], (1.0449, 1.070))
    assert lim == (376, 385)
