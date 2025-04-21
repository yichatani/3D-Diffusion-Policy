import h5py
import os
import shutil

ROOT_DIR = ["/home/ani/Dataset/episodes","/home/ani/astar/my_Isaac/episodes"] 
POSITIVE_DIR = os.path.join(ROOT_DIR[1], "positive")
NEGATIVE_DIR = os.path.join(ROOT_DIR[1], "negative")

def ensure_dir(path):
    """
    make sure the dir exists
    """
    if not os.path.exists(path):
        os.makedirs(path)

# Read structure
def read_structure(path):
    """
    Read the data structure of h5 files.
    """
    with h5py.File(path, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)
        f.visititems(print_hdf5_structure)


def read_values(path, key):
    """
    Read the key values. 
    """
    with h5py.File(path, 'r') as f:
        dset = f[key]
        print("Dataset shape:", dset.shape)
        print("First 10 values:", dset[:10])
        return dset[:]


def get_unique_filename(directory, filename):
    """
    if filename exist in root: file.h5 → file_1.h5, file_2.h5 ...
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def move_file_by_label(path):
    """
    Separate files by label: positive or negative, and avoid overwriting.
    """
    with h5py.File(path, 'r') as f:
        label = f["label"][()]

    label_value = label[0] if hasattr(label, '__len__') else label

    if label_value == 1:
        ensure_dir(POSITIVE_DIR)
        target_dir = POSITIVE_DIR
    elif label_value == 0:
        ensure_dir(NEGATIVE_DIR)
        target_dir = NEGATIVE_DIR
    else:
        print(f"Unexpected label value {label_value} in file: {path}")
        return

    filename = os.path.basename(path)
    unique_filename = get_unique_filename(target_dir, filename)
    target_path = os.path.join(target_dir, unique_filename)
    shutil.move(path, target_path)
    print(f"Moved {path} -> {target_path}")


def split_PN(root_dir):
    for filename in os.listdir(root_dir):
        if filename.endswith(".h5"):
            file_path = os.path.join(root_dir, filename)
            try:
                move_file_by_label(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


def count_h5_files(root_dir):
    i = 0
    for filename in os.listdir(root_dir):
        if filename.endswith(".h5"):
            i += 1
    print(f"file number: {i}")


if __name__ == "__main__":
    
    split_PN(ROOT_DIR[1])
    count_h5_files(POSITIVE_DIR)
    # count_h5_files(NEGATIVE_DIR)
    

    
    # path = POSITIVE_DIR + "/episode_1.h5"
    # read_structure(path)
    # print("======")
    # read_values(path,"label")
    # read_values(path, "agent_pos")
    # print("======")
    # read_values(path, "action")