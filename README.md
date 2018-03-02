# bme590hrm
Heart Rate Monitor Project for BME590

[![Build Status](https://travis-ci.org/edl13/bme590hrm.svg?branch=master)](https://travis-ci.org/edl13/bme590hrm)

## Intro
Welcome to my BME590 Heart Rate Monitor project.

## Use
Read the [docs](http://edward-liang-heart-rate-monitor.readthedocs.io/en)!

Import the module from HeartRateMonitor.py

```python
from HeartRateMonitor import HeartRateMonitor
```

Next, initialize the monitor object. The initalizer takes either a list or a csv filename.
Default units are ms and mV, but s and min, as well as V for voltage, can also be used.

```python
h = HeartRateMonitor(filename='data.csv', t_units='s', v_units='V')

# Run characterization functions
h.detect_bpm()
h.detect_voltage_extremes()
h.get_peaks()
h.get_duration()
```

Data can then be exported into JSON format. Units will be ms and mV.

```python
h.export_json()
```

