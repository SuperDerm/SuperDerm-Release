# SuperDerm-424K v0.1: Public Sources

Every photograph in SuperDerm-424K that comes from a public dataset: labels, splits and quality tiers for
395,432 clinical photographs (249,574 with a diagnosis) from 21 public datasets, mapped to the
214-diagnosis SuperDerm taxonomy. It accompanies the SuperDerm preprint.

**Images are not included.** Download them from each provider under its terms, then run `match_images.py` to
name them by `image_id`. **The NYU collection is not included**; it follows in a later release.

For research discussion only: nothing in this release, including any model trained on it, is medical advice or a
medical diagnosis.

## Contents

| File | What it holds |
|---|---|
| `images.csv` | One row per photograph (columns below) |
| `sources.csv` | Each source: row counts, licence or terms, homepage, citation and where to get the images |
| `diagnoses.v1.json` | The SuperDerm taxonomy: 214 diagnoses in 9 categories, with rare and high-consequence flags |
| `match_images.py` | Links downloaded files to rows by SHA-256, then by path (Python standard library only) |

## Counts

| Dataset | Rows | Train | Test | Unlabeled |
|---|---:|---:|---:|---:|
| Derm1M | 323,699 | 165,903 | 4,036 | 153,760 |
| SD-260 | 19,035 | 17,591 | 571 | 873 |
| Fitzpatrick17k | 12,398 | 11,697 | 660 | 41 |
| SCIN | 10,322 | 5,900 | 160 | 4,262 |
| MPXV-CNN | 9,826 | 0 | 0 | 9,826 |
| DermaCon-IN | 5,433 | 4,936 | 134 | 363 |
| PASSION | 3,645 | 2,935 | 50 | 660 |
| SkinCAP | 2,869 | 2,714 | 147 | 8 |
| PAD-UFES-20 | 2,282 | 2,257 | 25 | 0 |
| SkinDisNet | 1,864 | 1,848 | 16 | 0 |
| Islam6 | 823 | 740 | 83 | 0 |
| MSLD_v2.0 | 754 | 714 | 40 | 0 |
| MPox-Vision | 680 | 648 | 32 | 0 |
| DDI | 652 | 501 | 16 | 135 |
| MSID | 648 | 621 | 27 | 0 |
| MCSI | 299 | 289 | 10 | 0 |
| Ahsan2022 | 119 | 112 | 7 | 0 |
| MSLD_v1 | 51 | 15 | 0 | 36 |
| SD-198 | 27 | 0 | 0 | 27 |
| MCVSLD | 4 | 4 | 0 | 0 |
| UKHSA | 2 | 2 | 0 | 0 |

## Columns of images.csv

| Column | Meaning |
|---|---|
| `image_id` | Stable SuperDerm ID |
| `dataset` | Source dataset |
| `source_file` | The file's path inside the provider's own download |
| `sha256` | SHA-256 of the file as distributed by the provider |
| `split` | `train`, `test` or `unlabeled` |
| `group_side` | The side (`train` or `test`) of the patient, case or video group the image belongs to |
| `diagnosis_id`, `diagnosis` | SuperDerm diagnosis (`DX-…` in `diagnoses.v1.json`); empty when the source gives none |
| `category` | Taxonomy category of the diagnosis |
| `quality_tier` | Image quality, `Q1` (best) to `Q4` |
| `video_frame` | `1` for frames from educational videos (Derm1M), whose labels come from transcripts: weak labels |
| `case_id` | Images of the same case or patient, as identified by the source |

## Splits

- Splits come from SuperDerm curation manifest v3.3. **test** is class-balanced and drawn from whole
  patient, case and video groups, so no group appears on both sides. **train** is every other labelled image.
- **unlabeled** images have no diagnosis. Use them for self-supervised pre-training only where `group_side` is
  `train`, so nothing from a test group leaks in.
- Video frames never enter test.
- There is no validation split: carve one from train, grouped by `case_id`.
- The preprint's out-of-distribution test holds out SkinDisNet, DermaCon-IN and PASSION entirely. To follow that
  protocol, do not train on rows from these three datasets.

## Licences and Credit

Each row's label is shared under the terms of its source; the creators of every source agreed to their labels being
shared. Images come only from each provider, under the provider's terms. Several sources are non-commercial, so use
the release as a whole for non-commercial research. The SCIN licence forbids any attempt to re-identify the people
photographed. Cite every source you use:

- **Derm1M** (Yan et al., ICCV 2025): CC BY-NC 4.0. https://huggingface.co/datasets/redlessone/Derm1M. Images: Hugging Face (gated: agree to non-commercial research use).
- **Fitzpatrick17k** (Groh et al., CVPR Workshops 2021): CC BY-NC-SA 3.0 (annotations). https://github.com/mattgroh/fitzpatrick17k. Images: the url column of fitzpatrick17k.csv (original atlas sites); files are named <md5hash>.jpg.
- **SCIN** (Ward et al., JAMA Network Open 2024): SCIN Data Use License. https://github.com/google-research-datasets/scin. Images: the dx-scin-public-data Google Cloud Storage bucket.
- **PAD-UFES-20** (Pacheco et al., Data in Brief 2020): CC BY 4.0. https://data.mendeley.com/datasets/zr7vgbcyr2/1. Images: Mendeley Data (imgs_part_1 to imgs_part_3).
- **SkinDisNet** (Sultana et al., Data in Brief 2025): CC BY-NC 4.0. https://data.mendeley.com/datasets/yj3md44hxg/2. Images: Mendeley Data, versions 1 and 2 (their folder names differ).
- **SkinCAP** (Zhou et al., 2024): CC BY-NC-SA 4.0. https://huggingface.co/datasets/joshuachou/SkinCAP. Images: Hugging Face.
- **DDI** (Daneshjou et al., Science Advances 2022): Labels shared with the creators' permission; images under the DDI data use agreement. https://ddi-dataset.github.io/. Images: the DDI site, after accepting its agreement.
- **PASSION** (Gottfrois et al., MICCAI 2024): Labels shared with the creators' permission; images under the PASSION data public license. https://passionderm.github.io/. Images: the PASSION site, after accepting its licence.
- **DermaCon-IN** (Madarkar et al., NeurIPS 2025 Datasets and Benchmarks): Labels shared with the creators' permission; images under the provider's terms. Images: the dataset's creators.
- **SD-260** (Yang et al., IEEE TNNLS 2020): Labels shared with the creators' permission; images under the provider's terms. Images: the dataset's creators.
- **SD-198** (Sun et al., ECCV 2016): Labels shared with the creators' permission; images under the provider's terms. Images: the dataset's creators.
- **MSLD_v1** (Ali et al., 2022): Provider's terms. Images: the MSLD v1 release.
- **MSLD_v2.0** (Ali et al., Biomedical Signal Processing and Control 2024): Provider's terms. Images: the MSLD v2.0 release.
- **MSID** (Bala et al., Mendeley Data 2023): Provider's terms. https://data.mendeley.com/datasets/r9bfpnvyxr/6. Images: Mendeley Data.
- **MPXV-CNN** (Thieme et al., Nature Medicine 2023): Provider's terms. Images: the web sources listed by MPXV-CNN; source_file begins with the site name.
- **Ahsan2022** (Ahsan et al., 2022): Provider's terms. Images: the Monkeypox image data collection repository.
- **Islam6**: Provider's terms. Images: the monkeypox-dataset repository.
- **MPox-Vision**: Provider's terms. Images: the MonkeyPox Skin Image Dataset for Computer Vision.
- **MCSI**: Provider's terms. Images: the MCSI release.
- **MCVSLD**: Provider's terms. Images: the MCVSLD release.
- **UKHSA** (UK Health Security Agency): Crown copyright, Open Government Licence. https://www.gov.uk/guidance/mpox. Images: the UK Health Security Agency mpox guidance.

## Not in This Version

- Images (see above) and the NYU collection.
- Four files that are not clinical photographs (a demo picture and figures from code repositories) were removed.
- Labels are the sources' own diagnoses mapped to the taxonomy, not re-adjudicated by SuperDerm clinicians.

## Version

v0.1, 2026-11-25. Built from curation manifest v3.3. Please cite the SuperDerm
preprint (Fang et al., 2026) and each source you use.
