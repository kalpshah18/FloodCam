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

## Where else to get real data (researched 2026-10-28/29; none of these hosts were reachable from the sandbox)

Reachability was tested with a plain HTTPS request: every host in the table below was refused by the
sandbox egress policy. To use any of them, add the host under Allowed domains in the cloud
environment's Network access settings (or run the download on a machine without that restriction and
copy the files in). Licences are from the dataset pages as found by search; confirm on the record
before redistributing.

### A. Same kind of data as now (staff gauge or reference object + measured level) - highest value

| Source | What | Size / labels | Licence | Hosts to allow | Why it helps |
|---|---|---|---|---|---|
| **USGS NIMS + NWIS, more cameras** | 1,353 cameras are in the public bucket; 34 are staff-gauge cameras (`data/usgs/cameras.csv`), incl. 12 shallow overland-flow / edge-of-field sites (WI_AO1-3, WI_KD1-2, WI_MT1-3, WI_RS1-2, IN_Overland_Flow_North/West..., MI tile drain flume) | thousands of frames per site; label = USGS gage height | public domain | `api.waterdata.usgs.gov` (camera metadata, NIMS API), `waterservices.usgs.gov` (gage height). Images already reachable | More sites = more than one camera, which is the main weakness of the current study. Same OpenChannelFlow plate style as in_overland, so transfer and pooled training are realistic |
| **USGS image-derived water-level data release** (ScienceBase/data.usgs.gov, "time-lapse imagery, image-derived water level data and model archive for three USGS streamgages, WY2020") | hourly time-lapse + stage CSVs + R code, 3 Wisconsin gauges | hourly, one water year | public domain | `www.sciencebase.gov`, `data.usgs.gov` | Independent camera sites with stage, ready-made labels, no NWIS query needed |
| **USGS RISE imagery training set** (ScienceBase) | time-lapse stream-gauge imagery with matching stage and weather CSVs, 5 min to 1 h interval | several sites | public domain | `www.sciencebase.gov` | Weather labels allow a rain / fog failure analysis |
| **Tewkesbury river-camera dataset** (Data in Brief, DOI 10.1016/j.dib.2020.106338) | 4 river cameras, 2012 flood, level from surveyed reference points, cross-checked with Environment Agency gauges, hourly daylight | one flood, 4 cameras | CC BY | the journal / University of Reading repository (`centaur.reading.ac.uk`, `www.sciencedirect.com`, `doi.org`) | A real flood with a reference-point based label; another continent and camera type |
| **UK/Ireland calibrated river cameras** (Vandaele et al., Environmental Data Science 2023) | 95 cameras, 32,715 images, linked to gauges | large | check the paper | Cambridge Core / author repository | The largest camera-to-gauge set found; good for a pooled or pretrained model |
| **TU Dresden OPARA image-based water stage** | Raspberry Pi camera, water contour, 3D shore model, reference gauge | one site | CC BY 4.0 | `opara.zih.tu-dresden.de` | Different method family (contour + 3D model) to compare with |
| **CrowdHydrology staff-gauge photos** (HydroShare 79b25776...) | 141 staff-gauge photos with manually measured levels | 141 | CC BY | `www.hydroshare.org` | Citizen photos: viewpoint and lighting diversity for a robustness test |
| **Tidal-marsh gauge-camera data** (Dryad, doi 10.5061/dryad.2v6wwpzqk) | time-lapse + stage | one site | check record | `datadryad.org` | Tidal (non-flood) water: a negative/transfer case |

### B. Urban flood with a reference object (depth from submerged objects)

| Source | What | Licence | Hosts |
|---|---|---|---|
| **V-FloodNet data** (xmlyqing00) | urban flood videos with water-level annotation; code already cloned for reference | check HF card | `huggingface.co`, `drive.google.com` |
| **FloodDET** (Mendeley Data px48m7vwvf) | 10,262 images, COCO JSON, depth encoded as submersion level of people/bicycles/cars/buses | check record | `data.mendeley.com` |
| **Inundation2Depth** (PMC12765332) | multi-source urban floodwater depth dataset (labels derived from aerial imagery + LiDAR, not measured) | open access | `pmc.ncbi.nlm.nih.gov` and the data host named in the paper |
| **European Flood 2013** (cvjena/eu-flood-dataset) | relevance labels + bounding boxes for flood images; GitHub has metadata only | per-image licences | image hosts named in `metadata.json` |
| **Taxi-based flood depth set** (IWA J. Hydroinformatics 2025) | 537 images/videos of flooded taxis, tape-measured depth | not confirmed public | ask the authors |

### C. Water segmentation (for a scene-level mask model and the upload mode)

FloodNet (UAV, CDLA, Drive/Dropbox), AIFloodSense (470 aerial images, CC BY 4.0, IEEE DataPort,
`ieee-dataport.org`), Kaggle Flood Area Segmentation (290 images, `www.kaggle.com`), FloodIMG (Kaggle),
RIWA river water segmentation (Kaggle), ADE20K / COCO-Stuff (water, river, sea classes), SegFormer
weights `nvidia/mit-b0` and `-b1` (`huggingface.co`). The plate reader does not need them.

### D. Collect it yourself (best match to the research question)

`docs/RECORDING_PROTOCOL.md`: tray or controlled patch beside a marked pole, tape-measured depths
0, 2, 3, 5, 10, 20 cm, at least 3 scenes. This is the only source with tape-measured labels at the 3 cm
threshold and would be the strongest addition to the report.

### Hosts to allow, in priority order

1. `api.waterdata.usgs.gov`, `waterservices.usgs.gov` (many more labelled USGS sites)
2. `www.sciencebase.gov`, `data.usgs.gov` (ready-made USGS image + stage releases)
3. `huggingface.co` and its CDN (V-FloodNet data, SegFormer weights)
4. `data.mendeley.com`, `zenodo.org`, `datadryad.org`, `www.hydroshare.org`, `opara.zih.tu-dresden.de`

Steps: cloud environment menu in the session title bar -> Edit -> Network access -> add the hosts under
Allowed domains (https://code.claude.com/docs/en/cloud-environments#network-access).
