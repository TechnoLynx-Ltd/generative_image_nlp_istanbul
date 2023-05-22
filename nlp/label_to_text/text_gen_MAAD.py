# text generation from labels in MAAD_Face dataset describing faces with 30k instances
# MAAD_face is available at https://github.com/pterhoer/MAAD-Face/releases/tag/MAADFACE
# the used pre-trained model is MVP-data-to-text available at https://huggingface.co/RUCAIBox/mvp-data-to-text
# @article{tang2022mvp,
#   title={MVP: Multi-task Supervised Pre-training for Natural Language Generation},
#   author={Tang, Tianyi and Li, Junyi and Zhao, Wayne Xin and Wen, Ji-Rong},
#   journal={arXiv preprint arXiv:2206.12131},
#   year={2022},
#   url={https://arxiv.org/abs/2206.12131},
# }

# model and files are not included in the repo, you need to clone/download them

import numpy as np
import csv
from transformers import MvpTokenizer, MvpForConditionalGeneration

tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
labels = []
prompt = ''
out = []
tabular_labels = []
output_labels = []
CSV_file_path = open("MAAD_Face.csv")
MAAD_data = np.genfromtxt(CSV_file_path, dtype=None, delimiter=',', names=True)
for j in range(49):
    labels.append(MAAD_data.dtype.names[j].replace('_', ' '))
        
with open("Gen_text.csv", "w", newline="") as f:
    writer = csv.writer(f)
    output_labels.extend([labels[0], labels[1], 'Description'])
    for i in range(MAAD_data.shape[0]):
        out = [MAAD_data[i][0].decode()]
        out.append(MAAD_data[i][1])
        prompt = 'Describe the following data: '
        tabular_labels = [MAAD_data[i][2]]
        if MAAD_data[i][2] == 1:
            prompt += labels[2]
        else:
            prompt += 'Female'
        for j in range(3,49):
            tabular_labels.append(MAAD_data[i][j])
            if j != 47 and MAAD_data[i][j] == 1:
                prompt += ' | '
                prompt += labels[j]
        inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = model.generate(**inputs)
        model_out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        out.append(model_out[0])
        writer.writerows(out)