const HtmlWebpackPlugin = require("html-webpack-plugin");
const HtmlInlineScriptPlugin = require('html-inline-script-webpack-plugin');
const path = require("path");

module.exports =
{
    devServer:
    {
        static: "./dist",
    },
    entry: "./src/index.js",
    mode: "production",
    module:
    {
        rules:
        [
            {
                test: /\.css$/i,
                use: ["style-loader", "css-loader"],
            },
            {
                test: /\.(png|svg|jpg|jpeg|gif)$/i,
                type: "asset/inline",
            },
            {
                test: /\.(bin)$/i,
                type: "asset/inline",
                generator:
                {
                    dataUrl: content =>
                    {
                        const base64_text =
                            Buffer.from(content).toString("base64");

                        return spacify(base64_text);
                    }
                }
            },
            {
                test: /\.json$/i,
                type: "asset/source",
            },
            {
                test: /\.tsx?$/,
                use: "ts-loader",
                exclude: /node_modules/,
            },
        ],
    },
    output:
    {
        clean: true,
        filename: "main.js",
        library: "index",
        path: path.resolve(__dirname, "./dist"),
        publicPath: "",
    },
    plugins:
    [
        new HtmlWebpackPlugin(
        {
            template: "./src/index.html",
            inject: "head",
        }),
        new HtmlInlineScriptPlugin(),
    ],
    resolve:
    {
        extensions: [".tsx", ".ts", ".js"],
        alias:
        {
            "@tensorflow/tfjs$":
                path.resolve(__dirname, "./custom_tfjs/custom_tfjs.js"),
            "@tensorflow/tfjs-core$":
                path.resolve(__dirname, "./custom_tfjs/custom_tfjs_core.js"),
        }
    },
};

function spacify(text)
{
    const chunk_size = 10;
    let spacified_text = "";
    let i = 0;

    while ((i + 1) * chunk_size < text.length)
    {
        spacified_text += text.substr(i * chunk_size, chunk_size) + " ";
        ++i;
    }

    spacified_text += text.substr(i * chunk_size);

    return spacified_text;
}