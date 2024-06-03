import torch.nn as nn

# tied autoencoder using off the shelf nn modules
class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.decoder.requires_grad_(False)

    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = self.decoder(encoded_feats)
        return encoded_feats, reconstructed_output

    def weight_typing(self):
        # Weight Tying
        self.decoder.weight = nn.Parameter(self.encoder.weight.transpose(0, 1))

    def get_encoder_weights(self) :
        return self.encoder.weight
