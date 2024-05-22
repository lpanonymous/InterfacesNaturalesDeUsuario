import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

ie = Core()
devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

classification_model_xml = "C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase9/ovhello/model/v3-small_224_1.0_float.xml"
model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread(filename="C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase9/ovhello/data/image/intel_rnb.jpeg"), code=cv2.COLOR_BGR2RGB)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))

# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
plt.imshow(image);

result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

# Convert the inference result to a class name.
imagenet_classes = open("C:/Users/zS22000728/Documents/InterfacesNaturalesDeUsuario/clase9/ovhello/data/datasets/imagenet/imagenet_2012.txt").read().splitlines()

# The model description states that for this model, class 0 is a background.
# Therefore, a background must be added at the beginning of imagenet_classes.
imagenet_classes = ['background'] + imagenet_classes

plt.title(imagenet_classes[result_index])
plt.show()
