import os
import cv2
import numpy as np
from scipy.io import loadmat, savemat


def convert_training_images_to_numpy_array_for_each_class_color(pathname, no_of_images_in_each_class, mat_file_path):
    directory = os.fsencode(pathname)
    subdirectories = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory, dI))]
    print(subdirectories)

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

        mat_file_name = dir_name + ".mat"
        modi_mat_file_path = mat_file_path + "/" + mat_file_name
        modi_label = dir_name

        conv_img_matrix = np.asarray(per_class_img_matrix)
        print(np.shape(conv_img_matrix))
        savemat(modi_mat_file_path, mdict={modi_label: conv_img_matrix})


def convert_training_images_to_numpy_array_for_each_class_gray(pathname, no_of_images_in_each_class, mat_file_path):
    directory = os.fsencode(pathname)
    subdirectories = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory, dI))]
    print(subdirectories)

    for directory in subdirectories:
        per_class_img_matrix = []
        dir_name = os.fsdecode(directory)
        print("DIR NAME :" + dir_name)
        file_no = 1
        for file in os.listdir(pathname + dir_name):
            filename = os.fsdecode(file)
            print("FILE NAME :" + filename)
            src_file = pathname + dir_name + "/" + filename
            image = cv2.imread(src_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            per_class_img_matrix.append(gray)
            if file_no == no_of_images_in_each_class:
                break
            file_no += 1

        mat_file_name = dir_name + ".mat"
        modi_mat_file_path = mat_file_path + "/" + mat_file_name
        modi_label = dir_name

        conv_img_matrix = np.asarray(per_class_img_matrix)
        print(np.shape(conv_img_matrix))
        savemat(modi_mat_file_path, mdict={modi_label: conv_img_matrix})


def read_train_data_for_each_class(dir_path):
    for file in os.listdir(dir_path):
        filename = os.fsdecode(file)

        class_name = filename.split('.')[0]
        print(class_name)

        dataset_path = dir_path + "/" + filename
        data_set = loadmat(dataset_path)

        train_data = data_set[class_name]

        training_no_of_images_in_class = np.shape(train_data)[0]
        print(np.shape(train_data))

        i = 0
        while i < training_no_of_images_in_class:
            cv2.imshow(str(i + 1), train_data[i])
            cv2.waitKey(0)
            i += 1

    cv2.destroyAllWindows()


def convert_testing_images_to_numpy_array_gray(pathname, mat_file_name, label):
    directory = os.fsencode(pathname)

    all_classes_img_matrix = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print("FILE NAME :" + filename)
        src_file = pathname + "/" + filename
        image = cv2.imread(src_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_classes_img_matrix.append(gray)

    conv_img_matrix = np.asarray(all_classes_img_matrix)
    print(np.shape(conv_img_matrix))

    savemat(mat_file_name, mdict={label: conv_img_matrix})


def read_test_data(file_path):
    data_set = loadmat(file_path)

    test_data = data_set['test']

    no_of_test_images = np.shape(test_data)[0]

    i = 0
    while i < no_of_test_images:
        cv2.imshow(str(i+1), test_data[i])
        cv2.waitKey(0)
        i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    number_of_images_from_each_class = 3000
    # convert_training_images_to_numpy_array_for_each_class_gray(
    #     'D:/ML_Project/asl-alphabet/asl_alphabet_train/asl_alphabet_train/',
    #     number_of_images_from_each_class, 'D:/ML_Project/data_mat_files/gray/training')

    # read_train_data_for_each_class('D:/ML_Project/data_mat_files/gray/training/')

    convert_testing_images_to_numpy_array_gray("D:/ML_Project/asl-alphabet/asl_alphabet_test/asl_alphabet_test/",
                                               'D:/ML_Project/data_mat_files/gray/testing/testing.mat', 'test')
    read_test_data('D:/ML_Project/data_mat_files/gray/testing/testing.mat')
