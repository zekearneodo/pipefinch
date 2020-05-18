# pipefinch
Pipeline for neural data, to go from intan (rhd/rhs) or Neuropixels (SpikeGLX) to mountain/kilo/klusta data formats and sorted spikes.

In the Kilosort Branch, data recorded using Neuropixels 3B with SpikeGLX can be first converted to the kwik format, then sorted with Kilosort2, and produce rasters/units examples.

For now, this repository is meant to work in the systems setup in the Hahnloser Lab at Institute of Neuroinformatics, and documentation is incomplete.


To get a walk-through of the process for spike sorting, look at the notebook
/notebooks_sort/g4r4_sgl_sort_kilo-zpikezorter-merge-sort.ipynb

To get a walk-through of the alignment of spikes to singing and rasters, look at the notebook
notebooks_sort/g4r4_rasters_zpike-merge.ipynb
