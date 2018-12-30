import numpy as np


def build_conf_mat_from_file(ground_truth_file, predicted_file):
    confusion_matrix = np.zeros((8, 8), int)
    with open(ground_truth_file, 'r') as f_gt, open(predicted_file,
                                                    'r') as f_pd:
        for _, (line_gt, line_pd) in enumerate(zip(f_gt, f_pd)):
            label_gt = int(line_gt)
            label_pd = int(line_pd)
            confusion_matrix[label_gt][label_pd] += 1

    return confusion_matrix


def overall_accuracy(conf_mat):
    mat_diag = 0
    all_values = 0
    for row in range(8):
        for col in range(8):
            all_values += conf_mat[row][col]
            if row == col:
                mat_diag += conf_mat[row][col]
    if all_values == 0:
        all_values = 1
    return mat_diag / all_values


def intersection_over_union(conf_mat):
    mat_diag = np.diagonal(conf_mat)
    err_summed_by_row = np.sum(conf_mat, axis=0)
    err_summed_by_col = np.sum(conf_mat, axis=1)
    divisor = err_summed_by_col + err_summed_by_row - mat_diag
    divisor[mat_diag == 0] = 1
    return mat_diag / divisor


if __name__ == "__main__":
    conf_mat = build_conf_mat_from_file(
        r'TestEnv/synthetic_testset2_downsample_5000.labels',
        r'TestEnv/synthetic_testset2_downsample_5000_predicted.labels')
    # r'./toy_dataset/testset_direct.labels',
    # r'./toy_dataset/testset_restoring.labels')
    print(conf_mat)
    np.savetxt(
        "TestEnv/synthetic2_report.txt",
        # r'Results/exp5report.txt',
        conf_mat,
        fmt='%8d',
        delimiter='\t',
        header='Confusion matrix is')
    ov_acc = overall_accuracy(conf_mat)
    with open(
            "TestEnv/synthetic2_report.txt",
            # r'Results/exp5report.txt',
            'a') as f:
        f.write("overall accuracy is: ")
        f.write("{:1.3f}".format(ov_acc))
    print("ov_acc is: {:1.3f}".format(ov_acc))
    IoU = intersection_over_union(conf_mat)
    np.round(IoU, 3, IoU)
    with open(
            "TestEnv/synthetic2_report.txt",
            # r'Results/exp5report.txt',
            'a') as f:
        f.write("\nIoU is:\n")
        f.write(str(IoU))
    print("IoU is: ")
    print(IoU)