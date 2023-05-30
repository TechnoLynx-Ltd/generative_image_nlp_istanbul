import "./index.css"

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";

import model_config_json from "../graph_model_js/model.json";
import model_weights from "../graph_model_js/group1-shard1of1.bin";
import { Inline_model_handler } from "./Inline_model_handler";

import empty_img from "../resources/init_img.png";
import logo_img from "../resources/logo.png";

let model = null;

// Global variables
var age = null
var race = null
var hair_color = null
var face_shape = null
var brown_eyes = null

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

export function init()
{
    console.log("init() - called");

    document.getElementById("logo_img").src = logo_img
    document.getElementById("generated_image").src = empty_img
    document.getElementById("ageRange").value = 0;

    age = document.getElementById("ageRange").value;
    race = document.getElementById("race").value;
    hair_color = document.getElementById("hair_color").value;
    face_shape = document.getElementById("face_shape").value;
    brown_eyes = document.getElementById("Brown_Eyes").value;

    // load AI model
    load_model()

}

export function load_img()
{

}
export function pictureChange() {
    
    document.getElementById("generated_image").src = '../michael-1.jpg';
}

export function tabular_data_creation() {
    
}
export function interpolate_latents()
{

}