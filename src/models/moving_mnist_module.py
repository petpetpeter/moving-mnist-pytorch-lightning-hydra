from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy



class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = 12
        self.n_steps_past = 10
        self.n_steps_ahead = 10  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()

        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    @pl.data_loader
    def train_dataloader(self):
        train_data = MovingMNIST(
            train=True,
            data_root=self.path,
            seq_len=self.n_steps_past + self.n_steps_ahead,
            image_size=64,
            deterministic=True,
            num_digits=2)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True)

        return train_loader

    @pl.data_loader
    def test_dataloader(self):
        test_data = MovingMNIST(
            train=False,
            data_root=self.path,
            seq_len=self.n_steps_past + self.n_steps_ahead,
            image_size=64,
            deterministic=True,
            num_digits=2)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=True)

        return test_loader
