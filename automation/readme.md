## Workflow

| Tasks                         | Resource                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `ci: sentinel`                | [![hCAPTCHA Sentinel](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/sentinel.yaml/badge.svg?branch=main)](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/sentinel.yaml) |
| `ci: collector`               | [![hCAPTCHA Collector](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/collector.yaml/badge.svg)](https://github.com/QIN2DIM/hcaptcha-challenger/actions/workflows/collector.yaml) |
| `datasets: VCS, annoate`      | [#roboflow](https://app.roboflow.com/), [#model-factory](https://github.com/beiyuouo/hcaptcha-model-factory) |
| `model: ResNet - train / val` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/captcha-challenger/hcaptcha-model-factory/blob/main/automation/roboflow_resnet.ipynb) |
| `model: YOLOv8 - train / val` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QIN2DIM/hcaptcha-challenger/blob/main/automation/roboflow_yolov8.ipynb) |
| `model: upload, upgrade`      | [#objects](https://github.com/QIN2DIM/hcaptcha-challenger/tree/main/src), [#modelhub](https://github.com/QIN2DIM/hcaptcha-challenger/releases/tag/model) |
| `datasets: public, archive`   | [#roboflow-universe](https://universe.roboflow.com/qin2dim/), [#captcha-datasets](https://github.com/captcha-challenger/hcaptcha-whistleblower) |

## Quick start

1. Download datasets

    ```bash
    python datasets_downloader.py
    ```

2. Annotate the binary images

    labeling in the workspace `[PROJECT]/database2309/<diagnosed_label_name>`

3. Startup mini-workflow

    Add `mini_workflow:focus_flags` and run the following command:

    ```bash
    python mini_workflow.py
    ```
    Examples:

    ```python
    focus_flags = {
        # "motor_vehicle": "motor_vehicle2309",
        # "chess_piece": "chess_piece2309"
        # "robot": "robot2309",
        "dog": "dog2309"
    }
    ```

    - Copy from: `[PROJECT]/database2309/<diagnosed_label_name>`
    - Paste to: `[PROJECT]/data/<diagnosed_label_name>`
    - Output to: `[PROJECT]/model/<diagnosed_label_name>/<model_name[flag].onnx>`
    - [For Developer] Upload to:  [#modelhub](https://github.com/QIN2DIM/hcaptcha-challenger/releases/tag/model)

4. [For Developer] Update [#objects](https://github.com/QIN2DIM/hcaptcha-challenger/tree/main/src) and new commit with `fixed: #issue-id` to close related issue.