import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils.utils_projection as utils_projection
import utils.utils_segmentation as utils_segmentation



class ImageSegmentation:
    def __init__(self, predictor, instance_category,data_annotation, H, W):
        self.predictor = predictor
        self.instance_category = instance_category
        self.data_annotation = data_annotation
        self.H, self.W = H, W

    def get_mask_all(self, img_path, sam_seeds):
        category = sam_seeds.keys()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        mask_all = {}

        for obj, x_y in sam_seeds.items():
            input_point = np.array(x_y)

            if len(input_point) == 0:
                mask_all[obj] = []
                continue

            # if len(input_point) > 10:
            #     input_point = np.array(utils_projection.sort_by_distance(x_y, 10))

            input_label = np.ones(len(input_point))
            mask = utils_segmentation.get_mask(input_point, input_label, self.predictor)
            mask_all[obj] = mask[0]

        return mask_all

    def get_depth_order(self, mask_all, sam_seeds):
        category_order = {key: 0 for key in sam_seeds.keys()}
        category_key = list(category_order.keys())

        for i in range(len(category_key)):
            for j in range(i + 1, len(category_key)):
                A, B = mask_all[category_key[i]], mask_all[category_key[j]]

                if len(A) == 0 or len(B) == 0:
                    continue

                overlap = np.logical_and(A, B)

                if np.any(overlap):
                    top, bottom = utils_segmentation.get_the_top(sam_seeds, overlap, i, j, category_key, [self.H, self.W])
                    category_order[category_key[top]] = category_order[category_key[bottom]] + 1

        sorted_dict = dict(sorted(category_order.items(), key=lambda item: item[1]))
        return sorted_dict

    def segment_image(self, img_path, sam_seeds_by_category):
        image = cv2.imread(img_path)
        mask_all = self.get_mask_all(img_path, sam_seeds_by_category)
        depth_ordered_mask = self.get_depth_order(mask_all, sam_seeds_by_category)
        colored_mask = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for key in depth_ordered_mask:
            if len(mask_all[key]) ==0:
                continue
            utils_segmentation.create_colored_array(input_array=mask_all[key], target_color=self.instance_category[key], colored_array=colored_mask)

        return colored_mask


    def segment_all_images(self, img_dir, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for img_name in self.data_annotation:
            # try:
            outpath = os.path.join(out_dir, img_name)
            if os.path.exists(outpath):
                continue
            img_path = os.path.join(img_dir, img_name)
            colored_mask = self.segment_image(img_path, self.data_annotation[img_name])
            plt.imsave(outpath, colored_mask)
            # except:
            #     print("img_name: ", img_name)
