import os
import cv2
import numpy as np
from scipy.io import loadmat, savemat


def convert_training_images_to_numpy_array_for_all_classes(pathname, no_of_images_in_each_class, mat_file_name, label):
    directory = os.fsencode(pathname)
    subdirectories = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory, dI))]
    print(subdirectories)

    all_classes_img_matrix = []
    for directory in subdirectories:
        per_class_img_matrix = []
        dir_name = os.fsdecode(directory)
        print("DIR NAME :" + dir_name)
        file_no = 1
        for file in os.listdir(pathname + dir_name):
            filename = os.fsdecode(file)
            print("FILE NAME :" + filename)
            src_file = pathname + dir_name + "/" + filename
            image = cv2.imread(src_file, cv2.COLOR_RGB2BGR)
            per_class_img_matrix.append(image)
            if file_no == no_of_images_in_each_class:
                break
            file_no += 1

        all_classes_img_matrix.append(per_class_img_matrix)

    conv_img_matrix = np.asarray(all_classes_img_matrix)
    print(np.shape(conv_img_matrix))

    savemat(mat_file_name, mdict={label: conv_img_matrix})


def read_train_data(file_path):
    data_set = loadmat(file_path)

    train_data = data_set['train']

    training_no_of_classes = np.shape(train_data)[0]
    training_no_of_images_in_each_class = np.shape(train_data)[1]

    i = 0
    j = 0
    while i < training_no_of_classes:
        while j < training_no_of_images_in_each_class:
            cv2.imshow(train_data[i, j])
            cv2.waitKey(0)
            j += 1
        i += 1

    cv2.destroyAllWindows()


def convert_testing_images_to_numpy_array(pathname, mat_file_name, label):
    directory = os.fsencode(pathname)

    all_classes_img_matrix = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print("FILE NAME :" + filename)
        src_file = pathname + "/" + filename
        image = cv2.imread(src_file, cv2.COLOR_RGB2BGR)
        all_classes_img_matrix.append(image)

    conv_img_matrix = np.asarray(all_classes_img_matrix)
    print(np.shape(conv_img_matrix))

    savemat(mat_file_name, mdict={label: conv_img_matrix})


def read_test_data(file_path):
    data_set = loadmat(file_path)

    train_data = data_set['test']

    no_of_test_images = np.shape(train_data)[0]

    i = 0
    while i < no_of_test_images:
        cv2.imshow(train_data[i])
        cv2.waitKey(0)
        i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    number_of_images_from_each_class = 3000
    # convert_training_images_to_numpy_array_for_all_classes(
    #                                        "D:/ML Project/asl-alphabet/asl_alphabet_train/asl_alphabet_train/",
    #                                        number_of_images_from_each_class, 'D:/ML Project/training.mat', 'train')
    # read_train_data('D:/ML Project/training.mat')

    # convert_testing_images_to_numpy_array("D:/ML Project/asl-alphabet/asl_alphabet_test/asl_alphabet_test/",
    #                                       'D:/ML Project/testing.mat', 'test')
    # read_test_data('D:/ML Project/testing.mat')
