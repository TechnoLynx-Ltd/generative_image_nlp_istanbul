
cd website
call npm install

REM generate TensorFlow javascript
tensorflowjs_converter --weight_shard_size_bytes 20971520 --input_format keras --output_format tfjs_graph_model ../saved_model/encoder.hd5  graph_model_js/
call npx tfjs-custom-module --config custom_tfjs_config.json
call npx webpack
cd ..
echo.
echo.
echo If there were no errors, the bundled index file is in website/dist/