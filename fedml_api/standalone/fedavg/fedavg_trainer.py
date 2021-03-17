import copy
import logging
import time
import numpy as np
import random
import wandb
from torch.optim import lr_scheduler
from fedml_api.standalone.fedavg.client import Client
import torch
from fedml_experiments.standalone.fedavg.enhanceClient import EnhancedCLient
class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.model = model
        self.model.train()

        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif self.args.client_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)
        else:
            raise ValueError("unsupported client optimizer {}".format(self.args.client_optimizer))
        if self.args.lr_decay == 'cosine':
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.comm_round, eta_min=args.eta_min)
        else:
            self.lr_scheduler = None

        if self.args.server_optimizer == "sgd":
            self.server_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.server_lr)
        elif self.args.server_optimizer == 'adam':
            self.server_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.server_lr,
                                              weight_decay=self.args.wd, amsgrad=True)
        else:
            raise ValueError("unsupported server optimizer {}".format(self.args.server_optimizer))

        self.client_list = []
        #self.client_importance = [ 1e8] * args.client_num_in_total
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = EnhancedCLient(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, self.model, self.optimizer)
            self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if not self.args.use_client_IS:
            #   TODO select clients according to its importance and #samples
            if client_num_in_total == client_num_per_round:
                client_indexes = [client_index for client_index in range(client_num_in_total)]
            else:
                num_clients = min(client_num_per_round, client_num_in_total)
                np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            #logging.info("client_indexes = %s" % str(client_indexes))
            return client_indexes
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            candidates = np.random.choice(range(client_num_in_total), num_clients*20, replace=False)
            candidates_importances = self.get_clients_importances(candidates)
            selected_indexes = torch.multinomial(torch.tensor(candidates_importances), client_num_per_round, False).tolist()
            return [candidates[x] for x in selected_indexes]

    def get_clients_importances(self, client_candidates):
        w_global = self.model.state_dict()
        importances = []
        client = self.client_list[0]
        avg_client_sample_num = self.train_data_num_in_total / self.args.client_num_in_total
        for client_idx in client_candidates:
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            avg_loss = client.estimate_client_importance(w_global)
            importances.append(avg_loss*np.exp(0.1 * (client.get_sample_number() - avg_client_sample_num) ) )

        return importances

    def train(self):
        w_global = self.model.state_dict()
        stale_is_weight = w_global
        if self.args.slim_training:
            self.widths = self.args.slim_widths
            self.client_width_dict = {client_index: random.choice(self.widths) for client_index in range(self.args.client_num_in_total)}
        for round_idx in range(self.args.comm_round):
            
            #logging.info("################Communication round : {}".format(round_idx))

            w_locals, loss_locals = [], []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self.client_sampling(round_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)
            #logging.info("client_indexes = " + str(client_indexes))
            client_losses = []

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                if self.args.slim_training:
                    client.model.set_width(self.client_width_dict[client_idx])
                    client.model.slim(self.args.slim_channels)
                # train on new dataset
                if self.args.use_data_IS and self.args.stale_IS_weight:
                    w, loss = client.train(w_global, stale_is_weight)
                else:
                    w, loss = client.train(w_global)

                # if self.args.slim_training:
                #    w = client.model.trimmed_weights()
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                loss_locals.append(copy.deepcopy(loss))
                #logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))
                #avg_loss = client.estimate_client_importance()
                #self.client_importance[client_idx] = avg_loss * client.get_sample_number() /self.train_data_num_in_total
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # update global weights
            stale_is_weight = w_global

            #w_global = self.server_update(w_locals, w_global)

            if self.args.slim_training:
                #w_global = self.aggregate_nan(w_locals, w_global)
                #w_global = self.aggregate(w_locals)
                w_global = self.server_update(w_locals, w_global)
            else:
                w_global = self.aggregate(w_locals)
            # logging.info("global weights = " + str(w_glob))

            # print loss
            if round_idx % 1 == 0:
                loss_avg = sum(loss_locals) / len(loss_locals)
                logging.info('Round {:3d}, Average loss {:.3f} Loss variance: {:.3f} Learning rate:{:.5f}'.format(round_idx, loss_avg,
                                                                                                                  np.var(loss_locals),
                                                                                        self.lr_scheduler.get_lr()[0] if self.lr_scheduler else self.args.lr))

            if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                if self.args.slim_training:
                    if round_idx == self.args.comm_round - 1:
                        if 1.0 not in self.widths:
                            widths = self.widths + [1.0]
                        else:
                            widths = self.widths
                        for width in (widths):
                            logging.info('Testing using Width_mult:{}'.format(width))
                            self.model.load_state_dict(w_global)
                            self.model.set_width(width)
                            if self.args.slim_channels == 'random':
                                self.args.slim_channels = 'magnitude'
                            self.model.slim(self.args.slim_channels)
                            self.local_test_on_all_clients(self.model, round_idx)
                    else:
                        logging.info('Testing using Width_mult:{}'.format(max(self.widths)))
                        self.model.load_state_dict(w_global)
                        self.model.apply(lambda m: setattr(m, 'width_mult', max(self.widths)))
                        self.local_test_on_all_clients(self.model, round_idx)
                else:
                    self.model.load_state_dict(w_global)
                    self.local_test_on_all_clients(self.model, round_idx)


    def aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def aggregate_nan(self, w_locals, w_global):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        avg_num = training_num / len(w_locals)

        (sample_num, params) = w_locals[0]
        for k in params.keys():
            all_params = np.array([w[1][k].numpy()*w[0]/avg_num for w in w_locals ])
            avg_params = torch.tensor(np.nanmean(all_params, axis=0))
            is_nan = torch.isnan(avg_params)
            avg_params[is_nan] = w_global[k][is_nan]
            params[k] = avg_params
        return params

    def server_update(self, w_locals, w_global):


        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num

        (sample_num, gradient) = w_locals[0]
        for k in gradient.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    gradient[k] = (w_global[k] - local_model_params[k]) * w
                else:
                    gradient[k] += (w_global[k] - local_model_params[k] ) * w

        self.model.load_state_dict(w_global)
        for k, param in self.model.named_parameters():
            if k in gradient.keys():
                param.grad = gradient[k]

        if self.args.server_gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.server_gradient_clip)
        self.server_optimizer.step()
        w = copy.deepcopy(self.model.cpu().state_dict())
        return w

    def local_test_on_all_clients(self, model_global, round_idx):
        
        if self.args.dataset in ["stackoverflow_lr",  "stackoverflow_nwp"]:
            # due to the amount of test set, only abount 10000 samples are tested each round
            testlist = random.sample(range(0, self.args.client_num_in_total), 100)
            logging.info("################local_test_round_{}_on_clients : {}".format(round_idx, str(testlist)))
        else:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            testlist = list(range(self.args.client_num_in_total))

        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        client = self.client_list[0]
        
        for client_idx in testlist:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(model_global, False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(model_global, True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if self.args.dataset == "stackoverflow_lr":
                train_metrics['precisions'].append(copy.deepcopy(train_local_metrics['test_precision']))
                train_metrics['recalls'].append(copy.deepcopy(train_local_metrics['test_recall']))
                test_metrics['precisions'].append(copy.deepcopy(test_local_metrics['test_precision']))
                test_metrics['recalls'].append(copy.deepcopy(test_local_metrics['test_recall']))
                # due to the amount of test set, only abount 10000 samples are tested each round
                if sum(test_metrics['num_samples']) >= 10000:
                    break

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Pre": train_precision, "round": round_idx})
            wandb.log({"Train/Rec": train_recall, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)

        else:
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)
