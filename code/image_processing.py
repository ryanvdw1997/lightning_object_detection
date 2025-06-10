import cv2
import numpy as np
import os


def adjust_gamma(image, gamma=0.2):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

src_dir = "../data/images"




def main(gamma=1.0):
    for s in ['train', 'val']:
        current_folder = os.path.join(src_dir, s)
        for file in os.listdir(current_folder):
            if not file.lower().endswith((".jpg", ".png")):
                continue

            # 1) Load BGR
            img_bgr = cv2.imread(os.path.join(current_folder, file))

            # 3) Convert BGR → RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 4) Save out as a new JPEG (will be stored in RGB internally)
            #    cv2.imwrite expects BGR, so convert back for saving:
            img_out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # 5) Save the image to a processed folder with all processed images
            # cv2.imwrite(os.path.join('../data/frames/processed/all/images', file), img_out)
            # 6) Save the image to a processed folder with processed images broken out by video no.
            if not (os.path.exists(f'../data/images/{s}')):
                os.makedirs(f'../data/images/{s}')
            cv2.imwrite(os.path.join(f'../data/images/{s}', file), img_out)

if __name__ == '__main__':
    main()