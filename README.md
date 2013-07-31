# SlicedPy

SlicedPy (pronounced "Sliced Pie") takes your whole-house power meter
readings and estimates the energy consumed by your individual
appliances.  In other words, it produces an (estimated) itemised
energy bill using just a single, whole-house power meter.

This process is sometimes called:

* "nonintrusive load monitoring (NILM)"
* "nonintrusive appliance load monitoring (NIALM)"
* "smart meter disaggregation"

# Requirements

* Whole-house power meter readings sampled once every 10 seconds or faster.
* [Pandas](http://pandas.pydata.org/)
* [Power Data Analysis (PDA)
   toolkit](https://github.com/JackKelly/pda/)
