import os
import typing


class BaseLabeler:
    def __init__(self, data_dir, num_class: int = 2, labels: typing.List[str] = None):
        self.data_dir = data_dir
        self.num_class = num_class
        if labels:
            self.labels = labels
        else:
            self.labels = [f"class_{i}" for i in range(num_class)]

        if self.num_class != len(self.labels):
            raise ValueError(
                f"Number of classes ({self.num_class}) does not match number of labels ({len(self.labels)})"
            )

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

    def run(self):
        raise NotImplementedError
