import toml
import click
import deepfake_detection.train_model as tm

@click.command()
@click.option("--mode", default="Train", help="Train or Test")
@click.option("--model_name", default="meso4", help="Model to train. options: meso4, mesoinception4, resnet, vgg, xception")
@click.option("--train_path", default="deepfake_detection/data/train", help="Training data path")
@click.option("--val_path", default="deepfake_detection/data/val", help="Validation data path")
@click.option("--test_path", default="deepfake_detection/data/test", help="Test data path")
@click.option("--continue_train", default=False, help="Continue training a trained model?")
@click.option("--save_as", default="meso4.pkl", help="Trained model to sae as")
@click.option("--trained_model_path", default="trained_models/meso4.pkl", help="Trained model path to train again")
def run(mode, model_name, train_path, val_path, test_path, save_as, continue_train, trained_model_path):
    config = toml.load('config.toml')
    
    if mode.lower() == 'train':
        epochs = config['epochs']
        batch_size = config['batch_size']
        tm.train(model_name, train_path, val_path, save_as, continue_train, trained_model_path, epochs, batch_size)
    elif mode.lower() == 'test':
        pass
    else:
        print('Unknown mode. Terminating process')

if __name__ == "__main__":
    run()