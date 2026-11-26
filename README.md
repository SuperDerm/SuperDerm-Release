<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/superderm-lockup-reversed.svg">
    <img src="assets/superderm-lockup.svg" alt="SuperDerm" width="340">
  </picture>
</p>

<p align="center"><b>Open releases of SuperDerm: data, taxonomy, evaluation and models</b><br>
<a href="https://superderm.org">superderm.org</a></p>

## Releases

- **SuperDerm-424K v0.1 · Public Sources**: labels, splits and quality tiers for 395,432 clinical photographs from 21
  public datasets, mapped to 214 diagnoses. No images and no NYU data.
  [Download](https://github.com/SuperDerm/SuperDerm-Release/releases/tag/424k-v0.1) ·
  [Dataset card](datasets/superderm-424k/README.md)
- **Diagnostic Taxonomy v1**: 214 diagnoses in 9 categories, with rare and high-consequence designations.
  [taxonomy/diagnoses.v1.json](taxonomy/diagnoses.v1.json)
- **Community Scorer 1.0.0**: validates prediction files and scores them under the community protocol; Python
  standard library only. [evaluator/](evaluator/)
- **SuperDerm VLM and Full System**: will be open-sourced here. The final versions are in training.

## Cite

Fang C, Cao X, Talio A, Ye W, Porter CW, Rehg JM, Coffee M. SuperDerm: A Benchmark for Open-Ended Diagnosis of Common
and Rare Skin Diseases. Preprint, 2026. Please also cite each source dataset you use (see the dataset card).

## Jointly Organized by

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/organizers-dark.png">
    <img src="assets/organizers-light.png" width="800"
      alt="PediaMed AI Lab, Anthropic, Ti Kay, NYU Langone Health and the University of Illinois Urbana-Champaign">
  </picture>
</p>

---

For research discussion only: nothing here, including model outputs, is medical advice or a medical diagnosis.
