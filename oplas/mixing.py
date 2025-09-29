import torch


@torch.no_grad()
def mix_stems(
    stems_in,  # a full collection of (chunked) stems in the time domain. will grab a random subset
    first_stem=1,  # ignore the 0th stem which is the 'mix' from MUSDB
    static_mix=True,  # if True, use stems and mix as is
    debug=False,
):
    """
    here we actually mix inputs and encode them and embed them.
    """
    if static_mix:  # use predefined mix channel, do nothing else
        g_stems = stems_in[:, 1:, :, :]  # cut off the first stem, the 'mix'
        g_mix = stems_in[:, 0, :, :]  # the mix is the first stem [B, T, C
        return {"g_stems": g_stems, "g_mix": g_mix}

    device = stems_in.device

    stems_full = stems_in[:, first_stem:, :, :]  # cut off the first stem, the 'mix'
    B, S, T, C = stems_full.shape  # batch, stems, time, channels

    # nmix is a mask for the tracks to get a nice spread of different stems
    nmix = torch.randint(2, (B, S)).unsqueeze(-1).unsqueeze(-1).to(device)

    if debug:
        # print("nmix = ", nmix)
        breakpoint()

    stems = stems_full.to(device)  # put all the stems in a tensor

    # apply mask
    stems = stems * nmix

    # random gains for each stem [-1,1]
    gains = torch.rand(B, S).unsqueeze(-1).unsqueeze(-1).to(device)
    gains = 2 * gains - 1

    g_stems = stems * gains  # stems with random gains applied

    g_mix = torch.sum(g_stems, 1)  # sum along stems dimension

    return {"g_stems": g_stems, "g_mix": g_mix}


# TODO: replace with an Encoder model class from models.py
def do_encode(audio, encoder_model, model_choice="vggish", device="cuda", debug=False):
    """Ok here we're really really encoding"""
    if debug:
        print("\ndo_encode: encoder_model =", encoder_model)
    if "clap" in model_choice.lower():
        while len(audio.shape) < 3:
            audio = audio.unsqueeze(0)  # add batch and/or channel dims
        if audio.shape[-1] == 2:  # stereo to mono  TODO: make this more robust
            audio = audio.mean(dim=-1)
        encodings = encoder_model.get_audio_embedding_from_data(
            x=audio.to(device), use_tensor=True
        ).to(audio.dtype)
    elif "vggish" in model_choice.lower():
        encodings = encoder_model(audio)
    else:
        raise NotImplementedError
    return encodings


# def mix_and_encode_old(stems_full, encoder_model, encode_op,  model_choice='vggish', debug=False):
#     """OLD version
#     This performs the mixing AND the calling of the encoder (assuming it exists)
#     """
#     m = mix_stems(stems_full, debug=debug)
#     B, S, T, C = m['g_stems'].shape  # batch, stems, time, channels
#     ys = []  # TODO: could make ys a tensor instead of list, for consistency with other variables
#     for i in range(S):
#         y = encode_op(m['g_stems'][:,i,:,:], encoder_model, model_choice=model_choice, device=device)
#         ys.append(y)
#     y_mix = encode_op(m['g_mix'].to(device), encoder_model, model_choice=model_choice, device=device)  # encode the mix in the given model
#     return  m | {'ys':ys, 'y_mix':y_mix }


def mix_stems_single(stems_in, static_mix=True, debug=False):
    """Mix a lot of stems from a single batch"""
    if static_mix:
        g_mix = torch.sum(stems_in, dim=1)
        return {"g_stems": stems_in, "g_mix": g_mix}

    device = stems_in.device

    # TODO randomly mute some of the tracks
    B, S, T = stems_in.shape  # batch, stems, time, channels

    # Choose uniform number of unmuted per batch
    k = torch.randint(0, S + 1, (B,), device=device)

    # Boolean mask with random number of tracks being muted.
    # The masked tracks do not need to be shuffled because
    # the batched data is already shuffled.
    mask = torch.zeros((B, S), device=device, dtype=torch.float32)
    arange = torch.arange(S, device=device).expand(B, S)
    mask[arange < k.unsqueeze(1)] = 1.0
    mask = mask.unsqueeze(-1)

    stems_in = stems_in * mask  # apply mask

    # TODO randomly apply gain
    gains = torch.rand(B, S).unsqueeze(-1).to(device)
    # gains = 2 * gains - 1

    # scale to decibels
    gains = gains * -36.0
    gains = 10 ** (gains / 10.0)

    stems_in = stems_in * gains

    # TODO  analyze rms to adjust gain without worrying about clipping?
    # I don't think this is a big issue because the random gains will generally lower things enough

    # TODO randomly duplicate stems (need to capture doubling behavior!)
    g_mix = torch.sum(stems_in, dim=1)
    return {"g_stems": stems_in, "g_mix": g_mix}


def mix_and_encode(stems_full, encoder, static_mix=False, debug=False):
    """
    This performs the mixing AND the calling of the encoder (assuming it exists)
    """
    m = mix_stems(stems_full, static_mix=static_mix, debug=debug)
    ys = []  # TODO: could make ys a tensor instead of list, for consistency with other variables
    for i in range(m["g_stems"].shape[1]):
        with torch.no_grad():
            y = encoder.forward(m["g_stems"][:, i, :, :])
        ys.append(y)
    y_mix = encoder.forward(m["g_mix"])
    return m | {"ys": ys, "y_mix": y_mix}


def mix_latents(stems, encoder, static_mix=False, debug=False):
    """
    Takes latents, re-encodes them to audio and mixes them
    """
    # stems: [batch, latents, time]

    stems_audio = encoder.decode(stems, max_waveform_length=44100 * 6, max_batch_size=2)


def mix_single(stems, encoder, static_mix=True, debug=False):
    m = mix_stems_single(stems, static_mix=static_mix, debug=debug)

    ys = []

    # this isn't working
    for i in range(m["g_stems"].shape[0]):
        with torch.no_grad():
            y = encoder.forward(m["g_stems"][i])
        ys.append(y)

    y_mix = encoder.forward(m["g_mix"])
    return m | {"ys": ys, "y_mix": y_mix}
