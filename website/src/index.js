import "./index.css"

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";

import model_config_json from "../graph_model_js/model.json";
import model_weights from "../graph_model_js/group1-shard1of1.bin";
import { Inline_model_handler } from "./Inline_model_handler";

import { } from "../node_modules/bootstrap/dist/js/bootstrap.min.js"
import { latent_dict, idx_latent_vector, latent_keys, init_latent_dict } from "./latent_dictionary.js"

import empty_img from "../resources/init_img.png";
import logo_img from "../resources/logo.png";

const LATENT_DIM = 256;
const OUTPUT_IDX = 6;
const MAX_INTERPOL = 10;

// Global variables
var latent_vec_1 = null
var latent_vec_2 = null
let model = null;
var empty_image = null;

function unspacify(text)
{
    const chunk_size = 10;
    const space_chunk_size = chunk_size + 1;
    let result = "";
    let i = 0;

    while ((i + 1) * space_chunk_size < text.length)
    {
        result += text.substr(space_chunk_size * i, chunk_size);
        ++i;
    }

    result += text.substr(i * space_chunk_size);

    return result;
}

export async function load_model()
{
    console.log("Inline_model_handler() - called");
    let inline_model_handler =
        new Inline_model_handler(model_config_json, unspacify(model_weights));

    console.log("loadGraphModel() - called");
    model = await tf.loadGraphModel(inline_model_handler);
}

function set_canvas_image(canvas_id, img)
{
    var canvas = document.getElementById(canvas_id);
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    // var foo = Canvas2Image.saveAsPNG(canvas);
    // var img = canvas.toDataURL("image/png");
}

export function init()
{
    console.log("init() - called");

    init_latent_dict();

    // load AI model
    // tf.setBackend('gpu');
    console.log("TensorFlow Backend:", tf.getBackend());
    try {
        load_model()
    } catch (e) {
        console.log("Model load exception:", e);
    }

    // document.getElementById("logo_img").src = logo_img

    empty_image = new Image(256, 256);
    empty_image.src = empty_img;
    empty_image.onload = function () {
        set_canvas_image("generated_image_1", empty_image);
        set_canvas_image("generated_image_2", empty_image);
        set_canvas_image("interpolated_image", empty_image);
    }
}

// Generate image based on the latent vector in @input
//
export function load_img(input, canvas_id)
{
    // Expand current position to 4D b/c model input requirement
    //  console.log("Array: ", input);
    const input_tensor = tf.expandDims(tf.tensor(input), 0);
    // input_tensor.print();
    // console.log("Tensor: ", input_tensor);
    // console.log("Tensor: ", input_tensor.dataSync());

    // Model predicts score (shape:(1,2)) of current position
    const result = model.predict(input_tensor);
    // console.log("result: ", result);
    //  result[0].print()
    //  result[1].print()
    //  result[2].print()
    //  result[3].print()
    //  result[4].print()
    //  result[5].print()
    //  result[6].print()
    // result[OUTPUT_IDX].print()
    const scores = result[OUTPUT_IDX];//.arraySync();
    console.log("scores[", OUTPUT_IDX, "].shape: ", scores.shape);

    var canvas = document.getElementById(canvas_id);
    var ctx = canvas.getContext("2d");
    var image = ctx.createImageData(1, 1); // pixel image
    const gen_data = scores.dataSync();
    //  console.log("gen_data: ", gen_data);
    for (let y=0; y<scores.shape[1]; y++) {
        for (let x=0; x<scores.shape[2]; x++) {
            let b = gen_data[(y*scores.shape[2] + x)*3 + 0];
            let g = gen_data[(y*scores.shape[2] + x)*3 + 1];
            let r = gen_data[(y*scores.shape[2] + x)*3 + 2];
            // console.log("X:", x, " Y:", y, " R:", r, " G:", g, " B:", b);
            if (isNaN(r) || isNaN(g) || isNaN(b)) {
                image.data[0] = (x+2*y) % 256;    // Red
                image.data[1] = (x+3*y) % 256;    // Green
                image.data[2] = (x+5*y) % 256;    // Blue
            } else {
                image.data[0] = Math.max(0, Math.min(255, (r + 1)*127));    // Red
                image.data[1] = Math.max(0, Math.min(255, (g + 1)*127));    // Green
                image.data[2] = Math.max(0, Math.min(255, (b + 1)*127));    // Blue
            }
            image.data[3] = 255;                                // Alpha
            ctx.putImageData(image, x, y);
        }
    }

    //  console.log("scores: ", scores);
    // console.log("scores array: ", Array.from(scores.dataSync()));
    // scores.print();
}

// identify best fitting latent vector for UI options
//
function get_latent_vector(img_idx)
{
    let html_selects = ["race", "hair_style", "face_shape"];
    let html_checkboxes = ["Brown_Eyes", "Bushy_Eyebrows", "Smiling", "Big_Nose", "No_Beard", "High_Cheekbones", "Fully_Visible_Forehead"];

    let latent_selector = new Array(idx_latent_vector).fill(0);
    let filter_keys = new Array(idx_latent_vector).fill("");

    for (let i=0; i<html_selects.length; i++) {
        let key = html_selects[i];
        let elem = document.getElementById(key + img_idx);
        if (elem == null) {
            console.log("Select HTML element not defined:", key + img_idx);
            continue;
        }

        let value = parseInt(elem.value);
        if (value > 0) {
            let idx_array = latent_keys[key.replaceAll('_', '').toLowerCase()];
            console.log(key + img_idx, " -> ", value, idx_array);
            let idx = idx_array[value -1];
            latent_selector[idx] = 1;
            filter_keys[idx] = key;
        }
    }

    for (let i=0; i<html_checkboxes.length; i++) {
        let key = html_checkboxes[i];
        let elem = document.getElementById(key + img_idx);
        if (elem == null) {
            console.log("Checkbox HTML element not defined:", key + img_idx);
            continue;
        }

        if (elem.checked) {
            let idx = latent_keys[key.replaceAll('_', '').toLowerCase()];
            latent_selector[idx] = 1;
            filter_keys[idx] = key;
        }
    }

    // set of valid latent vector indices
    let valid_idxs = new Set();
    for (let i=0; i<latent_dict.length; i++) {
        valid_idxs.add(i);
    }

    for (let i=0; i<filter_keys.length; i++) {
        if (latent_selector[i] == 0) {
            // this feature is not part of the filter
            continue;
        }

        let found = false;
        for (const idx of valid_idxs) {
            if (latent_dict[idx][i] == 1) {
                found = true;
                break;
            }
        }
        if (!found) {
            console.log("Ignore filter:", filter_keys[i]);
            continue;
        }

        // We have at least one matching row so we can delete the others
        for (let idx of valid_idxs) {
            if (latent_dict[idx][i] != 1) {
                valid_idxs.delete(idx);
            }
        }
    }

    // return a random index from the indexes matching filter
    let rv = Array.from(valid_idxs);
    let rnd_idx = Math.floor(Math.random() * rv.length);
    console.log("latent_selector:", latent_selector);
    console.log("filter_keys:", filter_keys);
    console.log("selected_latent:", latent_dict[rv[rnd_idx]].slice(0, idx_latent_vector-1));
    console.log("Latent vectors matching filter: ", rv.length, " / ", rnd_idx, ' => LatentIdx: ', rv[rnd_idx]);
    //  console.log("Latent vector[", rnd_idx, "]: ", latent_dict[rv[rnd_idx]][idx_latent_vector]);
    return latent_dict[rv[rnd_idx]][idx_latent_vector];
}

// Handler functions for image generation
//
export function pic_gen_1()
{
    console.log("pic_gen_1");

    // randomly generated N = 40 length array 0 <= A[N] <= 39
    // let input = Array.from({length: LATENT_DIM}, () => Math.random());
    let input = get_latent_vector("_1");
    latent_vec_1 = input;
    load_img(input, "generated_image_1");

}

export function pic_gen_2()
{
    console.log("pic_gen_2");

    // randomly generated N = 40 length array 0 <= A[N] <= 39
    // let input = Array.from({length: LATENT_DIM}, () => Math.random());
    let input = get_latent_vector("_2");
    latent_vec_2 = input;
    load_img(input, "generated_image_2");
}

export function pic_gen_interpolate()
{
    console.log("pic_gen_interpolate");
    if ((latent_vec_1 == null) || (latent_vec_2 == null))
    {
        alert("Please generate the two input images first!");
        return;
    }

    let elem = document.getElementById("interpolate");
    console.log("Interpolation weight: ", parseInt(elem.value));
    let weight = parseInt(elem.value) / MAX_INTERPOL;

    // Interpolate the two latent vectors
    let input = new Array(LATENT_DIM).fill(0.0);
    for (let i=0; i<latent_vec_1.length; i++) {
        input[i] = (1 - weight) * latent_vec_1[i] + weight * latent_vec_2[i];
    }
    load_img(input, "interpolated_image");
}
