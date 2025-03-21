import h5py
import os
import shutil

ROOT_DIR = "/media/mldadmin/home/s124mdg32_04/episodes"
POSITIVE_DIR = os.path.join(ROOT_DIR, "positive")
NEGATIVE_DIR = os.path.join(ROOT_DIR, "negative")

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


def move_file_by_label(path):
    """
    separate files by label: positive or negative
    """
    with h5py.File(path, 'r') as f:
        label = f["label"][()]
    

    label_value = label[0] if hasattr(label, '__len__') else label

    if label_value == 1:
        ensure_dir(POSITIVE_DIR)
        shutil.move(path, os.path.join(POSITIVE_DIR, os.path.basename(path)))
        print(f"Moved {path} -> {POSITIVE_DIR}")
    elif label_value == 0:
        ensure_dir(NEGATIVE_DIR)
        shutil.move(path, os.path.join(NEGATIVE_DIR, os.path.basename(path)))
        print(f"Moved {path} -> {NEGATIVE_DIR}")
    else:
        print(f"Unexpected label value {label_value} in file: {path}")


if __name__ == "__main__":
    for filename in os.listdir(ROOT_DIR):
        if filename.endswith(".h5"):
            file_path = os.path.join(ROOT_DIR, filename)
            try:
                move_file_by_label(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    #     path = ROOT_DIR + "/episode_7.h5"
#     # read_structure(path)
#     print("======")
#     read_values(path,"label")
#     # read_values(path, "agent_pos")
#     # print("======")
#     # read_values(path, "action")