import ssn_dataset
import time

if __name__ == '__main__':
    start = time.time()
    csv_file = "~/Dataset/soft_shadow/single_human/metadata.csv"
    training_dataset = ssn_dataset.SSN_Dataset(csv_file, is_training=True)
    testing_dataset = ssn_dataset.SSN_Dataset(csv_file, is_training=False)

    print("Training dataset num: ", len(training_dataset))
    print("Testing dataset num: ",len(testing_dataset))

    for i in range(len(training_dataset)):
        data = training_dataset[i]
        print('Training set: successfully iterate {} \r'.format(i), flush=True, end='')

    for i in range(len(testing_dataset)):
        data = testing_dataset[i]
        print('Validation set: successfully iterate {} \r'.format(i), flush=True, end='')

    end = time.time()
