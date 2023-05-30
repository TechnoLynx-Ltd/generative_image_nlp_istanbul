import { } from "../models/decoder.js"
import { } from "../models/encoder.js"


var age = document.getElementById("ageRange").value;
var race = document.getElementById("race").value;
var hair_color = document.getElementById("hair_color").value;
var face_shape = document.getElementById("face_shape").value;
var brown_eyes = document.getElementById("Brown_Eyes").value;

function init()
{
    document.getElementById("generated_image").src = '../resources/init_img.png'
    document.getElementById("ageRange").value = 0;
}

function load_img()
{

}
function pictureChange() {
    
    document.getElementById("generated_image").src = '../michael-1.jpg';
}

function tabular_data_creation() {
    
}
function interpolate_latents()
{

}