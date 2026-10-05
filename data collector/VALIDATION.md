# Validation record

Validated on 2026-10-05. The collector's training-data scope is restricted to clips verified as recorded in India. A human reviewer must enter `country=India` and provide `location_evidence` before a source can be downloaded or a crash annotation can be created. Permission metadata separates rights-holder/license permission from a reviewer-asserted research basis. Research-basis records can only use the selected-segment downloader; this status records the reviewer's position and is not an automated legal determination. Foreign footage previously held as a separate experiment was removed from this workspace, including source videos and derived annotations/contact sheets. The Wikimedia/Commons ingestion route was removed.

The original Indian news pilot database retains 25 source-page candidates for audit. It contains no permitted downloads, no human-reviewed Indian clips and no unique verified Indian incidents. Page candidates and publisher metadata are not video files and do not count as verified Indian footage. No video bytes are retained in the collector dataset until India location and reuse permission are documented.

The gated Nexar repository was not accessed. YouTube discovery did not run because no API key was configured. The existing approximately 3,000 local clips were not imported because their folder was not specified.

UVH-26 by AIM @ IISc is documented as an optional second Indian-traffic data source for vehicle detection. It contains images and COCO bounding boxes, not crash videos; no UVH-26 files have been downloaded. Its Hugging Face card declares CC BY 4.0 and reports an unavailable dataset viewer due to schema metadata; check upstream before selecting files.

News-archive segment download is available through `download-segment`. Unit tests verify the requested range is passed to yt-dlp, overlong output is rejected, and selected-segment provenance is recorded. Actual website downloads were not run during this validation.

Run the test suite from `data collector`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
