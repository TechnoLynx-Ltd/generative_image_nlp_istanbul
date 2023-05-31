
cd website
call npm install

REM generate TensorFlow javascript
tensorflowjs_converter --weight_shard_size_bytes 20971520 --input_format keras --output_format tfjs_graph_model ../saved_model/decoder.h5  graph_model_js/
call npx tfjs-custom-module --config custom_tfjs_config.json

REM generate latent vector dictionary in JavaScrpt
call python ../vae/csv_2_js.py ./resources/latent.csv ./src/latent_dictionary.js

call npx webpack

cd ..
echo.
echo.
echo DONE. If there were no errors, the bundled index file is in website/dist/