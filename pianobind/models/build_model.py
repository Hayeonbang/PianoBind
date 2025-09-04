from pianobind.models.audio_text import PianoBind_AudioText
from pianobind.models.midi_text import PianoBind_MIDIText
from pianobind.models.trimodal import PianoBind_Trimodal


def create_model(model_config):
    model_name = model_config.model_name
    if model_name == "audio_text":
        return PianoBind_AudioText(model_config)
    elif model_name == "midi_text":
        return PianoBind_MIDIText(model_config)
    elif model_name == "trimodal":
        return PianoBind_Trimodal(model_config)
    else:
        raise ValueError(f"{model_name} is not supported.")


def freeze_modules(model, freeze_config, logger):
    if freeze_config.get("text_encoder", False):
        logger.write("Freeze the Text Encoder")
        if hasattr(model, "textual_head"):
            for param in model.textual_head.parameters():
                param.requires_grad = False

    if freeze_config.get("audio_encoder", False):
        logger.write("Freeze the Audio Encoder")
        if hasattr(model, "audio_backbone"):
            for param in model.audio_backbone.parameters():
                param.requires_grad = False

    if freeze_config.get("midi_encoder", False):
        logger.write("Freeze the MIDI Encoder")
        if hasattr(model, "midi_backbone"):
            for param in model.midi_backbone.parameters():
                param.requires_grad = False
