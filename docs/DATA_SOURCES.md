# Data sources: research, access status and what is used

Researched 2026-10-28. "Reachable" = reachable from the project's cloud sandbox, whose network policy
allows GitHub and a few package hosts only (huggingface.co, zenodo.org, kaggle.com, figshare,
ieee-dataport.org, opara.zih.tu-dresden.de, data.mendeley.com, api.waterdata.usgs.gov,
waterservices.usgs.gov, drive.google.com, dropbox.com were refused). Everything below that is marked
"used" was downloaded and is reproducible from the scripts in this repo.

## Used: USGS gauge cameras (real fixed-camera footage with official depth labels)

| item | detail |
|---|---|
| What | Still frames from USGS National Imagery Management System cameras pointed at staff gauges, every ~15 min to hourly, day, night (infrared), rain, glare, lens dirt |
| Labels | Official USGS gage height at the frame time (NWIS). A gauge sensor, not a camera estimate, is the label |
| Licence | US Government work, public domain. Attribution text in `data/usgs/README.md` |
| Frames | public S3 bucket `usgs-nims-images` (reachable); fetched by `scripts/build_usgs_dataset.py` |
| Truth tables | `data/usgs/truth/*.csv`, copied from the committed result tables of `tanayvasishtha/waterline` (Apache-2.0, commit a77a74c) and `Waqar-msx9t/stagesight` (MIT, commit e9b10429), which queried NWIS (not reachable here) |
| Site `in_overland` | USGS 411229084541101, Overland Flow near Harlan IN. Shallow sheet flow, plate 0-2 ft (0-62 cm). 2259 labelled frames, 15 monthly blocks 2023-2026, gauge 0-55 cm. **Primary dataset**: the shallow regime and the 3 cm (0.1 ft) threshold of this project |
| Site `hancock` | USGS 01421500, East Branch Delaware River at Hancock NY. River staff plate, 9-18 ft. 1992 labelled frames downloaded; mostly below the plate, one flood. Used for the dashboard demo/qualitative checks only (the dev blocks never exceed 12 ft, so a model cannot be trained for the July flood) |
| Limits | One camera per site: effective sample size is the number of monthly blocks (events/sessions), not frames. The gauge sits beside the plate (offset fitted on training blocks only). Depth below the plate bottom (~4 cm at in_overland) is not visible on the plate |

## Found, not usable here (blocked hosts), ranked by value if the network policy is widened

1. **V-FloodNet data** (urban flood videos with water-level annotation; HF `xmlyqing00/V-FloodNet`,
   Google Drive; code MIT-style at github.com/xmlyqing00/V-FloodNet, cloned for reference only).
2. **TU Dresden OPARA image-based water stage dataset** (low-cost camera + reference gauge + 3D shore
   model, CC-BY-4.0): opara.zih.tu-dresden.de.
3. **More USGS cameras**: `api.waterdata.usgs.gov` (camera lists) and `waterservices.usgs.gov`
   (gauge height) would add dozens of staff-gauge sites, including more shallow ones.
4. **Public water-segmentation sets** for a pretrained mask model: FloodNet (CDLA, UAV, Drive/Dropbox),
   AIFloodSense (CC BY 4.0, IEEE DataPort), Kaggle Flood Area Segmentation, ADE20K / COCO-Stuff, and
   SegFormer weights on huggingface.co. Not needed by the plate reader, which is trained from gauge
   labels; needed only for scene-level water masks.
5. European Flood 2013 dataset (github.com/cvjena/eu-flood-dataset): metadata and relevance labels
   only; images are hosted elsewhere and carry per-image licences.

To unlock (3) and (4) the environment's allowed-domains list needs those hosts (settings: cloud
environment menu -> Network access).
