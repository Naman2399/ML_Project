import torch

# To evaluate the network we will feed one character at a time,
# use the outputs of the network as a probability distribution for the next character,
# and repeat.

# To start generation we pass a priming string to start building up the hidden state,
# from which we then generate one character at a time.

def evaluate(model, dataset, start_words='A', predict_len=30, temperature=0.8, device = 'cpu'):

    hidden = model.init_hidden(batch_size= 1)
    hidden = hidden.to(device)
    prime_input = dataset.char_tensor(start_words)
    prime_input = prime_input.to(device)
    predicted = start_words

    # Use priming string to "build up" hidden state
    for p in range(len(start_words) - 1):
        input = prime_input[p]
        input = torch.reshape(input, (1, -1))
        _, hidden = model(input, hidden)
    input = prime_input[-1]
    input = torch.reshape(input, (1, -1))

    for p in range(predict_len):
        output, hidden = model(input, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = dataset.all_characters[top_i]
        predicted += predicted_char
        input = dataset.char_tensor(predicted_char)
        input = torch.reshape(input, (1, -1))
        input = input.to(device)

    return predicted