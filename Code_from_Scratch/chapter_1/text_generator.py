from model_loader import ModelLoader
import torch

class TextGenerator:

    def __init__(self):
        model_loader = ModelLoader()
        self.tokenizer=model_loader.load_tokenizer()
        self.model=model_loader.load_model()

    @torch.inference_mode()
    def generate_text(self,token_ids,max_new_tokens=100,eos_token_id=None):
        self.model.eval()

        for _ in range(self.max_new_tokens):

