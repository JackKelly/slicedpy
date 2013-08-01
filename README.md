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

# Why the name "SlicedPy"?

Smart meter disaggregation is a little like taking a pie (representing
your whole-home energy consumption) and slicing it into its component
pieces (each representing the energy consumed by an individual
appliance); hence the name "SlicedPy".  It's spelt "py" not "pie"
because the code is mostly written in Python.
