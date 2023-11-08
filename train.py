from option.option import BaseOptions
from trainer.style_trainer import StyleTrainer
def main() :
    opt = BaseOptions().parse()
    trainer = StyleTrainer(opt)
    trainer.fit()

if __name__ == "__main__" :
    main()