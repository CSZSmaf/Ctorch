#include "decoder_model.h"
#include <dl/onnx.h>

#include <iostream>

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::cerr << "usage: ctorch_export_decoder_onnx <weights.ct> <decoder.onnx>\n";
        return 1;
    }
    examples::ByteDecoder model;
    model.load_state_dict(dl::io::load(argv[1]));
    dl::onnx::save_decoder(argv[2], model.embedding(), model.decoder(), model.output_layer());
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
