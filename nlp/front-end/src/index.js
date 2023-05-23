function pictureChange() {
    document.getElementById("generated_image").src = '../michael-1.jpg';
    document.write(document.getElementById("race").value);
}
function tabular_data_creation() {
    var race = document.getElementById("race").value;
    document.write(race)
}