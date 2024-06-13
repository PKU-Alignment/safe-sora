<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
  <img src="docs/images/SafeSora.png" alt="SafeSora Logo" width="60%"/>
</div>
<h1 align="center">Towards Safety Alignment of Text2Video Generation </h1>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)](docs/CODE_LICENCE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-red.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)](docs/DATA_LICENCE)

<!-- [[`📕 Paper`](https://arxiv.org/abs/2307.04657)] -->
[[`🏠 Project Homepage`](https://sites.google.com/view/safe-sora)]
[[`🤗 SafeSora Datasets`](https://huggingface.co/datasets/PKU-Alignment/SafeSora)]
[[`🤗 SafeSora Label`](https://huggingface.co/datasets/PKU-Alignment/SafeSora-Label)]
[[`🤗 SafeSora Evaluation`](https://huggingface.co/datasets/PKU-Alignment/SafeSora-Eval)]
[[`BibTeX`](#citation)]

SafeSora is a human preference dataset designed to support safety alignment research in the text-to-video generation field, aiming to enhance the helpfulness and harmlessness of Large Vision Models (LVMs). It currently contains three types of data:

- A [classification dataset](#multi-label-classification-dataset) of 57k+ Text-Video pairs, including multi-label classification of 12 harm labels for their text prompts and text-video pairs.
- A [human preference dataset](#human-preference-dataset) of 51k+ instances in the text-to-video generation task, containing comparative relationships in terms of helpfulness and harmlessness, as well as four sub-dimensions of helpfulness.
- An [evaluation dataset](#evaluation-dataset) containing 600 human-written prompts, with 300 being safety-neutral and another 300 constructed according to 12 harm categories as red-team prompts.

In the future, we will also open-source some baseline alignment algorithms that utilize these datasets.

<!-- ## What's New -->

### Table of Contents  <!-- omit in toc --> <!-- markdownlint-disable heading-increment -->

- [Dataset Release](#dataset-release)
  - [Multi-label Classification Dataset](#multi-label-classification-dataset)
  - [Human Preference Dataset](#human-preference-dataset)
  - [Evaluation Dataset](#evaluation-dataset)
- [Data Access](#data-access)
- [Citation](#citation)
- [License](#license)

## Dataset Release

### Multi-label Classification Dataset

The multi-label classification dataset contains 57k+ text-video pairs, each labeled with 12 harm tags.
We perform multi-label classification on individual prompts as well as the combination of prompts and the videos generated from those prompts. These 12 harm tags are defined as:

- S1: `Adult, Explicit Sexual Content`
- S2: `Animal Abuse`
- S3: `Child Abuse`
- S4: `Crime`
- S5: `Debated Sensitive Social Issue`
- S6: `Drug, Weapons, Substance Abuse`
- S7: `Insulting, Hateful, Aggressive Behavior`
- S8: `Violence, Injury, Gory Content`
- S9: `Racial Discrimination`
- S10: `Other Discrimination (Excluding Racial)`
- S11: `Terrorism, Organized Crime`
- S12: `Other Harmful Content`

<!-- For more details, please see here. -->

The distribution of these 14 categories is shown below:

<div align="center">
  <img src="docs/images/data_ratio.png" alt="Data Ratio" width="85%"/>
</div>

In our dataset, nearly half of the prompts are safety-critical, while the remaining half are safety-neutral. Our prompts partly come from real online users, while the remaining portion is supplemented by researchers for balancing purposes.

For more information, please refer to **Hugging Face Page**: [PKU-Alignment/SafeSora-Label](https://huggingface.co/datasets/PKU-Alignment/SafeSora).
<!-- - **Data Card**: [SafeSora Label](data/SafeSora-Label). -->

### Human Preference Dataset

The human preference dataset contains over 51,000 comparisons, each data point comprising a user input and two generated videos. Through the following heuristic-based annotation process, human preferences were obtained in terms of `helpfulness` or `harmlessness` dimensions.

Additionally, due to a pre-annotation process, human preferences on four helpfulness sub-dimensions were also included. These sub-dimensions are:

- `Instruction Following`
- `Correctness`
- `Informativeness`
- `Aesthetics`

The specific annotation process is as shown in the figure below:

<div align="center">
  <img src="docs/images/annotation_pipeline.png" alt="Annotation Process" width="85%"/>
</div>

For more information, please refer to **Hugging Face Page**: [PKU-Alignment/SafeSora](https://huggingface.co/datasets/PKU-Alignment/SafeSora).

### Evaluation Dataset

The evaluation dataset contains 600 human-written prompts, including 300 safety-neutral prompts and 300 red-teaming prompts. The 300 red-teaming prompts are constructed based on 12 harmful categories. These prompts will not appear in the training set and are reserved for researchers to generate videos for model evaluation.

For more information, please refer to **Hugging Face Page**: [PKU-Alignment/SafeSora-Eval](https://huggingface.co/datasets/PKU-Alignment/SafeSora-Eval).

## Data Access

The dataset is available on the Hugging Face Datasets Hub.
A recommended way to download is using huggingface cli:

```bash
# Multi-label Classification Dataset: SafeSora-Label
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False --resume-download PKU-Alignment/SafeSora-Label --local-dir ./SafeSora-Label

# Human Preference Dataset: SafeSora
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False --resume-download PKU-Alignment/SafeSora --local-dir ./SafeSora

# Evaluation Dataset: SafeSora-Eval
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False --resume-download PKU-Alignment/SafeSora-Eval --local-dir ./SafeSora-Eval
```

The downloaded data mainly consists of two parts: `config-train.json.gz` and `config-test.json.gz` are the data configurations, and `videos.tar.gz` is the compressed package of videos. Please unzip the package before use.

```bash
tar -xzvf video.tar.gz
```

Each data point in the dataset includes a user prompt, the potential harmful category of the user prompt, a generated video, and the annotation results of the harmful category for the Text-Video pair. In the config, the video will include a `video_path` pointing to its relative location in the videos folder. This relative location follows a fixed rule: `videos/prompt_id/video_id`.

**Note:** The `videos.tar.gz` file in the SafeSora-Label and SafeSora preference datasets is the same, so if you have previously downloaded `videos.tar.gz`, you can use the same video folder and only need to download the config files separately.

We also provide a script to quickly return a Torch Dataset class:

```python
from safe_sora.datasets import VideoDataset, PairDataset, PromptDataset

# Multi-label Classification Dataset
label_data = VideoDataset.load("path/to/config", video_dir="path/to/video_dir")

# Human Preference Dataset
pref_data = PairDataset.load("path/to/config", video_dir="path/to/video_dir")

# Evaluation Dataset
eval_data = PromptDataset.load("path/to/config", video_dir="path/to/video_dir")
```

## Citation

If you find the SafeSora dataset family useful in your research, please cite the following paper:

```bibtex
@article{SafeSora2024,
  title   = {SafeSora: Towards Safety Alignment of Text2Video Generation via a Human Preference Dataset},
  author  = {Josef Dai and Tianle Chen and Xuyao Wang and Ziran Yang and Taiye Chen and Jiaming Ji and Yaodong Yang},
  url     = {https://github.com/calico-1226/safe-sora},
  year    = {2024}
}
```

<!-- ## Acknowledgment  omit in toc -->

## License

SafeSora dataset and its family are released under the CC BY-NC 4.0 License.
The code is released under Apache License 2.0.
