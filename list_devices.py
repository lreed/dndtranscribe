import sounddevice as sd

print("Available audio devices:\n")
devices = sd.query_devices()
for i, dev in enumerate(devices):
    marker = ""
    if dev["max_input_channels"] > 0 and dev["max_output_channels"] > 0:
        marker = " [IN/OUT]"
    elif dev["max_input_channels"] > 0:
        marker = " [IN]"
    elif dev["max_output_channels"] > 0:
        marker = " [OUT]"

    highlight = "  >>> " if "cable output" in dev["name"].lower() else "      "
    print(f"{highlight}{i:3d}: {dev['name']}{marker}  (in={dev['max_input_channels']}, out={dev['max_output_channels']}, {dev['default_samplerate']:.0f}Hz)")

print("\nLook for 'CABLE Output' with [IN] — that's the device your transcription script reads from.")
