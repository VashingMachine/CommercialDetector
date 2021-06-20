import onnxruntime as rt
from image_dataset import TestDataset
sess = rt.InferenceSession("polsat_detector.onnx")
hard_dataset = TestDataset('/home/ktoztam/CLionProjects/CommercialDetector/classifier/data/test')

for i in range(10):
    image, _ = hard_dataset[i]
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: image.unsqueeze(0).numpy()})
    print(pred)
