#include "decoder_model.h"
#include "text_training.h"

int main(int argc, char** argv) {
    return examples::train_text<examples::ByteDecoder>(argc, argv, examples::decoder_text);
}
