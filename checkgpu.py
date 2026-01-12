import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if not gpus:
    print("❌ No GPU found")
else:
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print("Device name:", details.get("device_name", "Unknown"))
        print("Compute capability:", details.get("compute_capability", "Unknown"))
        print("-" * 40)
