from googlenet_arch import MnistClassifier

class Experiments:

    def get_models(self):
        mc = MnistClassifier()

        # model_name = ['f-32_fs-[3_3]_a-relu_co-adam_cl-categorical_crossentropy_fe-10_fbs-64']
        # filters = [32]
        # filter_size = [(3, 3)]
        # activation = ['relu']
        # compile_optimizer = ['adam']
        # compile_loss = ['categorical_crossentropy']
        # fit_epochs = [10]
        # fit_batch_size = [64]

        # for i in range(len(model_name)):
        #     mc.train_model(model_name[i], filters[i], filter_size[i], activation[i], compile_optimizer[i], compile_loss[i], fit_epochs[i], fit_batch_size[i])

        # GoogLeNet
        model_names = ['googlenetFirst']
        compile_optimizers = ['adam']
        compile_losses = ['categorical_crossentropy']
        fit_epochs = [10]
        fit_batch_sizes = [64]

        # Iteracja po modelach i ich parametrach do treningu
        for i in range(len(model_names)):
            mc.train_model(model_names[i], compile_optimizers[i], compile_losses[i], fit_epochs[i], fit_batch_sizes[i])


if __name__ == "__main__":
    experiments = Experiments()
    experiments.get_models()