import torch
from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Tokenizer,Qwen3Model,QWEN_CONFIG_06_B
from pathlib import Path


class ModelLoader:
    # function simply used for the downloading of Qwen tokenizer and the pre-trained model for the project
    def __init__(self):
        self.out_dir="qwen3"
        self.out_dir_model="qwen3_model"
        self.tokenizer=None
        self.model=None
        self.device=None
        download_qwen3_small(kind="base",tokenizer_only=True,out_dir=self.out_dir)
        download_qwen3_small(kind="base",tokenizer_only=False,out_dir=self.out_dir_model)
        self.tokenizer_path=Path(self.out_dir) / "tokenizer-base.json"
        self.model_path=Path(self.out_dir_model) / "qwen3-0.6B-base.pth"


    # checking the availability of GPU in the system
    def check_gpu(self, enable_tensor_cores=True):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            if enable_tensor_cores:
                major , minor= map(int, torch.__version__.split(".")[:2])

                if(major,minor)>=(2,11):
                    torch.backends.cuda.matmul.fp32_precision="tf32"
                    torch.backends.cudnn.conv.fp32_precision="fp32"

                else:
                    torch.backends.cuda.matmul.fp32_precision=True
                    torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")

        return self.device

    # instantiating the tokenizer
    def load_tokenizer(self):
        self.tokenizer = Qwen3Tokenizer(tokenizer_file_path=self.tokenizer_path)

        # test run to check if the tokenizer is working

        prompt="wow the first run"
        print(self.tokenizer.encode(prompt))

        return self.tokenizer

    def load_model(self):
        self.model=Qwen3Model(QWEN_CONFIG_06_B)
        self.model.load_state_dict(torch.load(self.model_path))

        if self.device=="cuda:0":
            self.model.to(self.device)

        return self.model














if __name__ == "__main__":
    ModelLoader().load_tokenizer()

