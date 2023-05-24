import os
import argparse
from random import random
from tqdm import tqdm
import re
import csv


parser = argparse.ArgumentParser()
parser.add_argument("--metadata_path", default="../../VGGface2_HQ/MAAD_Face.csv")
parser.add_argument("--images_path", default="../../VGGface2_HQ/VGGface2_None_norm_512_true_bygfpgan")
parser.add_argument("--out_path", default="../../VGGface2_HQ/dataset")
args = parser.parse_args()

# TODO: take only 2-3 images per person, use it as parameter
def check_row(row):
    person_id, fname = row[0].split("/")
    # if this person's id even exists in our image set
    if os.path.isdir(os.path.join(args.images_path, person_id)):
        fnames_images = os.listdir(os.path.join(args.images_path, person_id))
        if fname in fnames_images:
            # csv row had a match in image set
            print("this tabul")
            # TODO: load image and move it to dataset, same name not sure about directories for different people
            return True
    return False


person_ids = os.listdir(args.images_path)
filtered_rows = []
with open(args.metadata_path, newline='') as csvfile:
    same_person_cntr = 0

    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    header = next(reader)
    for row in reader:
        # skip column names
        print(', '.join(row))
        print("--------------")
        print(row[0])
        if check_row(row):
            filtered_rows.append(row)
            same_person_cntr += 1
            # TODO: skip to next person if cntr reaches 2 or 3


with open(args.out_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(filtered_rows)

print("done")
#
# aug_val = 0.5
# if args.force_cpu:
#     print("Forced training on CPU")
# else:
#     ModelLoader.load_cuda()
#
# input_fnames = os.listdir(args.input_dir)
# print("Length of dataset is: " + str(len(input_fnames)))
#
#
# A_counter = 0
# B_counter = 0
# aug_counter = 0
# for i in tqdm(range(len(input_fnames))):
#     fname = input_fnames[i]
#     img = DataManager.load_style_pytorch(os.path.join(args.input_dir, fname), [512, 512])
#     regex_string = "(A|B).jpg$"
#     output = re.search(regex_string, fname)
#     match = output.group()
#
#     # file name to save with (remove A or B from the end:
#     new_fname = fname.replace(match, ".jpg")
#
#     if "A" in match:
#         pair_name = fname.replace(match, match.replace("A", "B"))
#         if os.path.exists(os.path.join(args.input_dir, pair_name)):
#             DataManager.save_file(img, os.path.join(args.label_a, new_fname))
#             A_counter += 1
#     elif "B" in match:
#         pair_name = fname.replace(match, match.replace("B", "A"))
#         if os.path.exists(os.path.join(args.input_dir, pair_name)):
#             DataManager.save_file(img, os.path.join(args.label_b, new_fname))
#             B_counter += 1
#
#             # If B image augment with a given percentage:
#             epsilon = random()
#             if epsilon < aug_val:
#                 fname_augmented = "aug_" + new_fname
#                 DataManager.save_file(img, os.path.join(args.label_a, fname_augmented))
#                 DataManager.save_file(img, os.path.join(args.label_b, fname_augmented))
#                 aug_counter += 1
#     else:
#         raise Exception("No label was found in image: " + fname)
#
#
# print("Augmented " + str(aug_counter) + " images, which is " + str((aug_counter/(len(input_fnames)/2))*100) + "% of the dataset")
# assert A_counter == B_counter
#
# a_fnames_len = len(sorted(os.listdir(args.label_a)))
# b_fnames_len = len(sorted(os.listdir(args.label_b)))
#
# assert a_fnames_len == b_fnames_len
#
# a_fnames = sorted(os.listdir(args.label_a))
# b_fnames = sorted(os.listdir(args.label_b))
# for i in range(len(a_fnames)):
#     assert a_fnames[i] == b_fnames[i]
#
# # This last assert just means there were images filtered out because they had no pairs.
# # sum = A_counter + B_counter
# # assert sum == len(input_fnames)
