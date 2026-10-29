# USGS gauge-camera dataset (real footage)

Real fixed-camera frames of staff gauges from the USGS National Imagery Management System
(public domain, https://api.waterdata.usgs.gov/docs/nims) paired with the official USGS gage height
(NWIS, public domain) for the same timestamp.

* Frames: public S3 bucket `usgs-nims-images` (`overlay/<camera>/<camera>___<UTC time>.jpg`).
  Fetched by `scripts/build_usgs_dataset.py`; stored under `data/usgs/frames/` (gitignored).
* Ground truth: `truth/*.csv` (`file,time,gage_ft`, committed, ~4k rows). These rows were taken
  from the committed result tables of two open-source projects that queried NWIS, because the NWIS
  hosts are not reachable from this environment:
  * `tanayvasishtha/waterline` (Apache-2.0), commit a77a74c, `results/*eval*_{in_overland,hancock}*.csv`
  * `Waqar-msx9t/stagesight` (MIT), commit e9b10429, `results/hancock_plate/series.csv`
  The `gage_ft` column is the USGS sensor value; no camera-derived number is used as a label.
* Sites: `in_overland` (USGS 411229084541101, shallow overland flow near Harlan IN, plate 0-2.1 ft,
  i.e. 0-64 cm: the regime of this project) and `hancock` (USGS 01421500, river staff plate, 9-21 ft).
* Only a few frames per site were visually checked; labels are the gauge, not hand annotation.
  The gauge sits next to, not on, the plate, so a small offset between plate reading and gauge is
  expected and is fitted on training events only.
