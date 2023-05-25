import os
import argparse
import csv
import cv2
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--metadata_path", default="../../VGGface2_HQ/MAAD_Face.csv")
parser.add_argument("--images_path", default="../../VGGface2_HQ/VGGface2_None_norm_512_true_bygfpgan")
parser.add_argument("--out_path", default="../../VGGface2_HQ/dataset", help="Output path, folder will be created automatically.")
parser.add_argument("--max_count", default=5000)
parser.add_argument("--img_per_id", default=3)
args = parser.parse_args()


def check_row(row, filtered_rows):
    person_id, fname = row[0].split("/")
    if os.path.isdir(os.path.join(args.images_path, person_id)):
        fnames_images = os.listdir(os.path.join(args.images_path, person_id))
        if fname in fnames_images:
            img_path_src = os.path.join(args.images_path, person_id, fname)
            new_fname = person_id + "_" + fname
            img_path_dst = os.path.join(args.out_path, "images", new_fname)
            row[0] = new_fname
            filtered_rows.append(row)
            img = cv2.imread(img_path_src)
            cv2.imwrite(img_path_dst, img)
            return True
    return False


assert os.path.isfile(args.metadata_path)
assert os.path.isdir(args.images_path)
if os.path.isdir(args.out_path):
    shutil.rmtree(args.out_path, ignore_errors=True)
Path(args.out_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.out_path, "images")).mkdir(parents=True, exist_ok=True)

global_cntr = 0
person_ids = os.listdir(args.images_path)
filtered_rows = []
with open(args.metadata_path, newline='') as csvfile:
    same_id_cntr = 0
    prev_id = ""
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    header = next(reader)
    line_count = args.max_count
    for row in reader:
        person_id, fname = row[0].split("/")
        if person_id != prev_id:
            same_id_cntr = 0
        if same_id_cntr >= args.img_per_id:
            continue
        if global_cntr >= args.max_count:
            break
        if check_row(row, filtered_rows):
            print(f"Global counter at: {global_cntr}")
            same_id_cntr += 1
            prev_id = person_id
            global_cntr += 1

print("Saving new csv.")
with open(os.path.join(args.out_path, "metadata.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header row
    writer.writerows(filtered_rows)

print("Done generating dataset.")
