from fedml_api.standalone.fedavg.client import Client
import logging, copy
import torch
from data_sampler import DataImportanceSampler, SplitSet
import torch.nn.functional as F

class EnhancedCLient(Client):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model,
                 optimizer):
        super(EnhancedCLient, self).__init__(client_idx, local_training_data, local_test_data, local_sample_number,
                                             args, device, model)

        self.optimizer = optimizer
        self.data_IS = args.use_data_IS
        self.alpha= args.data_IS_alpha
        self.loss = -1
        self.width_mult = 1.0


    def _weight_add(self, a, b, alpha=0.5):

        assert 0 <= alpha <= 1

        c = copy.deepcopy(a)
        for k in c.keys():
            c[k] = c[k].cpu() * alpha + b[k] * (1- alpha)
        return c



    def train(self, w_global, is_w=None):
        self.model.train()
        self.model.load_state_dict(w_global)
        self.model.to(self.device)
        if self.data_IS:
            importance_model = copy.deepcopy(self.model)
            d = SplitSet(self.local_training_data.dataset)
            sampler = DataImportanceSampler(d)
            train_loader = torch.utils.data.DataLoader(d, sampler=sampler, batch_size=self.local_training_data.batch_size, drop_last=False)
        else:
            train_loader = self.local_training_data

        epoch_loss = []
        for epoch in range(self.args.epochs):
            if self.data_IS:
                w = is_w if is_w else self.model.state_dict()
                corrected_w = self._weight_add(w, w_global, alpha=self.alpha)
                importance_model.load_state_dict(corrected_w)
                sampler.update_importance(importance_model, device=self.device, smooth_factor=0.01)

            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                self.model.zero_grad()

                if self.args.slim_training and self.args.multi_losses>0:
                    losses = []
                    for width in self.args.slim_widths:
                        if width<=self.width_mult:
                            self.model.set_width(width)
                            self.model.slim(self.args.slim_channels)
                            log_probs = self.model(x)
                            loss = self.criterion(log_probs, labels)
                            if width<self.width_mult:
                                loss = loss*self.args.multi_losses
                            loss.backward()

                            # if width<self.width_mult:
                            #     losses.append(self.args.multi_losses*loss)
                            # else:
                            #     losses.append(loss)
                    #loss = torch.sum(torch.stack(losses))
                else:
                    log_probs = self.model(x)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

        return self.model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)


    def estimate_client_importance(self, w):
        self.model.load_state_dict(w)
        self.model.eval()
        self.model.to(self.device)
        loss_list = []
        for batch_idx, (x, labels) in enumerate(self.local_training_data):

            # compute sample importances
            with torch.no_grad():
                data = x.to(self.device)
                label = labels.to(self.device)
                output = self.model(data)
                losses = F.cross_entropy(output, label, reduction='none')
                loss_list.append(losses)

        avg_loss = torch.mean(torch.cat(loss_list))

        return avg_loss