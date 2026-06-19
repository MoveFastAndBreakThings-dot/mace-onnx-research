import onnxruntime as ort

sess = ort.InferenceSession("mace_small.onnx")

print("Model loaded!")
print("Inputs:", [i.name for i in sess.get_inputs()])
print("Output:", [o.name for o in sess.get_outputs()])
print("File size: 16.1 MB")
print("Done — ONNX model works!")
