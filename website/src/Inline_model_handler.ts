import * as tf from "@tensorflow/tfjs";

// Mainly copied from
// https://github.com/tensorflow/tfjs/blob/9cb4d53fa20dd58349a131c834c3054906dbb897/tfjs-node/src/io/file_system.ts#L165
export class Inline_model_handler implements tf.io.IOHandler {
    protected readonly model_json: tf.io.ModelJSON;
    protected readonly weight_data: ArrayBuffer;

    constructor(model_json_string: string, base64_weights: string)
    {
        this.model_json = JSON.parse(model_json_string);
        this.weight_data = base64_to_ArrayBuffer(base64_weights);
    }

    async load(): Promise<tf.io.ModelArtifacts>
    {
        const model_json: tf.io.ModelJSON = this.model_json;

        const modelArtifacts: tf.io.ModelArtifacts =
        {
            modelTopology: model_json.modelTopology,
            format: model_json.format,
            generatedBy: model_json.generatedBy,
            convertedBy: model_json.convertedBy
        };

        if (model_json.weightsManifest != null)
        {
            const weight_specs: tf.io.WeightsManifestEntry[] = [];

            for (const group of model_json.weightsManifest)
            {
                weight_specs.push(...group.weights);
            }

            modelArtifacts.weightSpecs = weight_specs;
            modelArtifacts.weightData = this.weight_data;
        }

        if (model_json.trainingConfig != null)
        {
            modelArtifacts.trainingConfig = model_json.trainingConfig;
        }

        if (model_json.userDefinedMetadata != null)
        {
            modelArtifacts.userDefinedMetadata = model_json.userDefinedMetadata;
        }

        return modelArtifacts;
    }
}

// https://stackoverflow.com/questions/21797299/convert-base64-string-to-arraybuffer
function base64_to_ArrayBuffer(base64: string) {
    var binary_string = window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array(len);

    for (var i = 0; i < len; i++)
    {
        bytes[i] = binary_string.charCodeAt(i);
    }

    return bytes.buffer;
}