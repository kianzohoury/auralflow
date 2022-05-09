
def print_conv_stats(model: nn.Module):
    for layer in model.parameters():
        if isinstance(layer, nn.Conv)