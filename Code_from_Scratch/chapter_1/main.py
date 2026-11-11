from model_loader import ModelLoader



def main():
    loader=ModelLoader()
    # device=loader.check_gpu()
    # loader.load_tokenizer()


    loader.load_model()


if __name__=="__main__":
    main()
