import os

import cv2

from config import config

# Deprecated, I hope you know that.

if __name__ == "__main__":
    data = os.listdir(config.data_path)

    f = open(config.label_file_path, "w")
    for i, img_name in enumerate(data):
        img_path = os.path.join(config.data_path, img_name)

        if (
            os.path.isdir(img_path)
            or img_name.startswith(".")
            or img_name.endswith("txt")
            or img_name.endswith("zip")
        ):
            continue

        img = cv2.imread(img_path)
        cv2.imshow("img", img)
        k = cv2.waitKey(0)
        if k >= ord("0") and k <= ord("9"):
            f.write(img_name + "," + str(k - ord("0")) + "\n")
            cv2.imwrite(os.path.join(config.labeled_data_path, str(k - ord("0")), img_name), img)
            print("{}/{} {}: {}".format(i + 1, len(data), img_name, k - ord("0")))
        elif k == ord("q"):
            print("Quit")
            break
        else:
            print("Drop img:", img_name)
            continue
    f.close()
