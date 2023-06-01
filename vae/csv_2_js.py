import sys
import pandas as pd
import array
import numpy as np

idx_count = 0
latent_array = []
latent_keys = {}

def add_idx(df, js, col_name):
    global idx_count
    global latent_keys

    latent_keys[col_name]     = df.columns.get_loc(col_name)
    latent_array.append(df.columns.get_loc(col_name))
    js.write("export let idx_" + col_name + " = " + str(idx_count) + ";     // " + col_name + "\n")
    idx_count += 1

def add_idx_array(df, js, name, col_names):
    global idx_count
    global latent_keys

    latent_keys[name]   = []
    for cn in col_names:
        latent_keys[name].append(df.columns.get_loc(cn))
    latent_array.append(latent_keys[name])
    js.write("export let idx_" + name + " = [" + ",".join(str(idx_count + i) for i in range(len(latent_keys[name]))) + "];      // " + str(col_names)+ "\n")
    idx_count += len(latent_keys[name])

def main(csv_name, js_name):
    global idx_count
    global latent_keys
    global latent_array

    df = pd.read_csv(csv_name, sep=',')
    js = open(js_name, "w")

    js.write("//////////////////////////////////////////\n")
    js.write("// !!! GENERATED CODE => DO NOT MODIFY !!!\n")
    js.write("//////////////////////////////////////////\n\n\n")

    js.write("// Indexes for the CSV columns\n//\n")

    add_idx(df, js, "Male")
    add_idx_array(df, js, "Age", ['Young', 'Middle_Aged', 'Senior'])
    add_idx_array(df, js, "Race", ['Asian', 'White','Black'])
    add_idx(df, js, "Rosy_Cheeks")
    add_idx(df, js, "Shiny_Skin")
    add_idx_array(df, js, "Face_Shape", ['Oval_Face', 'Square_Face', 'Round_Face'])
    add_idx_array(df, js, "Hair_Style", ['Bald', 'Wavy_Hair', 'Black_Hair', 'Receding_Hairline', 'Bangs'])
    add_idx(df, js, "Sideburns")
    add_idx_array(df, js, "HairColor", ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    add_idx(df, js, "No_Beard")
    add_idx_array(df, js, "Facial_Hair", ['Mustache', '5_o_Clock_Shadow', 'Goatee'])
    add_idx(df, js, "Double_Chin")
    add_idx(df, js, "High_Cheekbones")
    add_idx(df, js, "Chubby")
    add_idx(df, js, "Obstructed_Forehead")
    add_idx(df, js, "Fully_Visible_Forehead")
    add_idx(df, js, "Brown_Eyes")
    add_idx(df, js, "Bags_Under_Eyes")
    add_idx(df, js, "Bushy_Eyebrows")
    # add_idx_array(df, js, "Eyebrow", ['Bushy_Eyebrows', 'Arched_Eyebrows'])
    # add_idx_array(df, js, "Mouth", ['Mouth_Closed', 'Smiling'])
    add_idx(df, js, "Smiling")
    add_idx(df, js, "Big_Lips")
    add_idx(df, js, "Big_Nose")
    add_idx_array(df, js, "Nose", ['Big_Nose', 'Pointy_Nose'])
    add_idx(df, js, "Heavy_Makeup")
    add_idx(df, js, "Wearing_Hat")
    add_idx(df, js, "Wearing_Earrings")
    add_idx(df, js, "Wearing_Necktie")
    add_idx(df, js, "Wearing_Lipstick")
    add_idx_array(df, js, "Glass", ['No_Eyewear', 'Eyeglasses'])
    add_idx(df, js, "Attractive")

    idx_latent_vector = df.columns.get_loc("latent_vector")
    js.write("export let idx_latent_vector = " + str(idx_count) + ";\n")
    idx_count += 1

    js.write("\nexport let latent_keys = {};\n")
    js.write("export let latent_dict = [];\n")

    js.write("\nexport function init_latent_dict()\n{\n")
    for key in latent_keys.keys():
        js.write("    latent_keys['" + key + "'.replaceAll('_', '').toLowerCase()] = idx_" + key + ";\n")

    # iterate over CSV lines
    js.write("\n// Latent vector dictionary\n//\n")
    for l in range(len(df)):
        js_line = "    latent_dict.push(["
        first = True
        for i in latent_array:
            if hasattr(i, "__len__"):
                for k in i:
                    if not first:
                        js_line += ", "
                    first = False
                    js_line += str(df.iloc[l, k])
            else:
                if not first:
                    js_line += ", "
                first = False
                js_line += str(df.iloc[l, i])
        js_line += "\n              , "
        latent_vector = str(df.iloc[l, idx_latent_vector])
        latent_vector = latent_vector.replace("\n", " ")
        latent_vector = latent_vector.replace("\r", " ")
        latent_vector = latent_vector.replace("\t", " ")
        while latent_vector.find("  ") != -1:
            latent_vector = latent_vector.replace("  ", " ")
        latent_vector = latent_vector.replace(" ", ", ")
        latent_vector = latent_vector.replace("[[,", "")
        latent_vector = latent_vector.replace(",]]", "")
        latent_vector = latent_vector.replace("[", "")
        latent_vector = latent_vector.replace("]", "")
        js_line += "[" + latent_vector + "]]);\n"
        js.write(js_line)

    js.write("}\n")

if __name__ == "__main__":
    if (len(sys.argv)) < 3:
        print("Usage: ", sys.argv, " input_csv_file output_js_file")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

