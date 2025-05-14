import numpy as np

def get_mask(input_point, input_label,predictor):
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False)
    return masks
def get_the_top(image_data,overlap,i,j,category_key,h_w):
    def count_points(points, overlap_np):
        cnt = 0
        h,w = h_w
        # print("h,w: ", h,w)
        for pixel in points:
            y,x = pixel
            # print("y,x: ", y,x)
            if 0<x<w  or 0< y<h:
                if overlap_np[x,y]:
                    # result.append(pixel)
                    cnt += 1
        return cnt
    overlap_np =overlap # np.array(overlap[0].cpu())
    count_i = count_points(np.array(image_data[category_key[i]]), overlap_np)
    count_j = count_points(np.array(image_data[category_key[j]]), overlap_np)
    
    return (i,j) if (count_i> count_j) else (j,i)
def create_colored_array(input_array, target_color,colored_array):
    h, w = input_array.shape
    # Find the indices where the input_array is True
    true_indices = np.where(input_array)

    # Fill the colored_array with the target_color at the True indices
    # colored_array[:, true_indices[0], true_indices[1]] = target_color
    for i in range(h):
        for j in range(w):
            if input_array[i,j] == True:
                colored_array[i,j,:] = target_color

    return colored_array
