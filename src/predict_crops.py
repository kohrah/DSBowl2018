import numpy as np
import cv2


def get_mirror_image_by_index(image, index):
    if index < 4:
        image = np.rot90(image, k=index)
    else:
        if len(image.shape) == 3:
            image = image[::-1, :, :]
        else:
            image = image[::-1, :]
        image = np.rot90(image, k=index-4)
    return image


def get_mirror_image_by_index_backward(image, index):
    if index < 4:
        image = np.rot90(image, k=-index)
    else:
        image = np.rot90(image, k=-(index-4))
        if len(image.shape) == 3:
            image = image[::-1, :, :]
        else:
            image = image[::-1, :]
    return image


def get_mask_from_model_v1(model, image, input_size=320, n_classes=2):
    from keras import backend as K
    from keras.applications.vgg16 import preprocess_input

    box_size = input_size
    initial_shape = image.shape
    initial_rows, initial_cols = image.shape[0], image.shape[1]

    print('Initial image shape: {}'.format(image.shape))
    if image.shape[0] < box_size or image.shape[1] < box_size:
        new_image = np.zeros((max(image.shape[0], box_size), max(image.shape[1], box_size), image.shape[2]))
        new_image[0:image.shape[0], 0:image.shape[1], :] = image
        image = new_image
        print('Rescale image... New shape: {}'.format(image.shape))

    if n_classes > 1:
        final_mask = np.zeros((initial_rows, initial_cols, n_classes), dtype=np.float32)
    else:
        final_mask = np.zeros(image.shape[:2], dtype=np.float32)

    count = np.zeros(image.shape[:2], dtype=np.float32)
    image_list = []
    params = []

    # 224 cases
    if 1:
        size_of_subimg = input_size
        # step = 14
        step = size_of_subimg // 8 #28
        for j in range(0, image.shape[0], step):
            for k in range(0, image.shape[1], step):
                start_0 = j
                start_1 = k
                end_0 = start_0 + size_of_subimg
                end_1 = start_1 + size_of_subimg
                if end_0 > image.shape[0]:
                    start_0 = image.shape[0] - size_of_subimg
                    end_0 = image.shape[0]
                if end_1 > image.shape[1]:
                    start_1 = image.shape[1] - size_of_subimg
                    end_1 = image.shape[1]

                image_part = image[start_0:end_0, start_1:end_1].copy()
                # cv2.imshow('p', image_part)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


                # FIXME FIXME PLEASE

                for i in range(4):
                    im = get_mirror_image_by_index(image_part.copy(), i)
                    # im = cv2.resize(im, (box_size, box_size), cv2.INTER_LANCZOS4)
                    image_list.append(im)
                    params.append((start_0, start_1, size_of_subimg, i))

                if k + size_of_subimg >= image.shape[1]:
                    break
            if j + size_of_subimg >= image.shape[0]:
                break

    print('Masks to calc: {}'.format(len(image_list)))
    image_list = np.array(image_list, dtype=np.float32)
    # image_list = preprocess_input(image_list)

    mask_list = model.predict(image_list, batch_size=48)
    # for mask in mask_list:
    #     predicted, pred_bounds, pred_seeds = cv2.split(mask)
    #     cv2.imshow('p', predicted)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # print('Number of masks:', mask_list.shape)

    border = 20
    for i in range(mask_list.shape[0]):
        if K.image_dim_ordering() == 'th':
            mask = mask_list[i, n_classes, :, :].copy()
        else:
            mask = mask_list[i, :, :, :].copy()
        if mask_list[i].shape[0] != params[i][2]:
            mask = cv2.resize(mask, (params[i][2], params[i][2]), cv2.INTER_LANCZOS4)
        mask = get_mirror_image_by_index_backward(mask, params[i][3])
          # show_resized_image(255*mask)

        # Find location of mask. Cut only central part for non border part
        if params[i][0] < border:
            start_0 = params[i][0]
            mask_start_0 = 0
        else:
            start_0 = params[i][0] + border
            mask_start_0 = border

        if params[i][0] + params[i][2] >= final_mask.shape[0] - border:
            end_0 = params[i][0] + params[i][2]
            mask_end_0 = mask.shape[0]
        else:
            end_0 = params[i][0] + params[i][2] - border
            mask_end_0 = mask.shape[0] - border

        if params[i][1] < border:
            start_1 = params[i][1]
            mask_start_1 = 0
        else:
            start_1 = params[i][1] + border
            mask_start_1 = border

        if params[i][1] + params[i][2] >= final_mask.shape[1] - border:
            end_1 = params[i][1] + params[i][2]
            mask_end_1 = mask.shape[1]
        else:
            end_1 = params[i][1] + params[i][2] - border
            mask_end_1 = mask.shape[1] - border

        final_mask[start_0:end_0, start_1:end_1] += \
            mask[mask_start_0:mask_end_0, mask_start_1:mask_end_1]
        count[start_0:end_0, start_1:end_1] += 1

    if count.min() == 0:
        print('Some uncovered parts of image!')
    # print(final_mask.shape)
    # print(np.max(final_mask), np.min(final_mask))

    predicted, pred_seeds = cv2.split(final_mask) / count
    # cv2.imshow('pr', pred_bounds)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    final_mask /= np.stack((count, count), 2)
    if initial_shape[:2] != final_mask.shape[:2]:
        final_mask = final_mask[0:initial_shape[0], 0:initial_shape[1]]
        print('Return shape back: {}'.format(final_mask.shape))

    # show_resized_image(255 * final_mask)
    return predicted, pred_seeds
    # return final_mask
