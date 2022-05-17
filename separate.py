import torch


from torch import Tensor


def separate(model, audio: Tensor):
    # Split audio in chunks.
    padding = model.dataset_params["hop_length"] // 2
    step_size = model.dataset_params.num_stft_frames
    max_frames = audio.shape[-1]

    # Store chunks.
    chunks = []

    offset = 0
    while offset < max_frames:
        # Reshape and trim audio chunk.
        audio_chunk = audio[:, offset : offset + step_size]

        # Unsqueeze batch dimension if not already batched.
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.unsqueeze(0)

        # Separate audio.
        estimate = model.separate(audio_chunk)

        # Trim end by padding amount.
        estimate = estimate[..., :-padding]
        chunks.append(estimate)

        # Update current frame position.
        offset = offset + step_size - padding

    # Stick chunks to create full source estimate.
    full_estimate = torch.cat(
        chunks,
        dim=0,
    )
    return full_estimate


if __name__ == "__main__":
    pass
