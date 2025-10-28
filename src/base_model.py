import onnxruntime as ort

class ModelLoader:
    """ Base class for loading ONNX models and performing inference """

    def __init__(self, onnx_path, device):
        self.onnx_path = onnx_path
        self.device = device

        self.session = ort.InferenceSession(
            onnx_path,
            providers = ['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, inputs):
        """ Run inference on the input data """
        
        return self.session.run(self.output_names, {self.input_name: inputs})