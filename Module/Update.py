
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """
        Needed for len()
        """
        return len(self.idxs)

    def __getitem__(self, item):
        """
        Needed for DatasetSplit[item]
        """
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, settings, dataset, indexes):
        self.settings = settings
        self.device = settings.device
        self.criterion = self.settings.criterion 
        self.criterion.to(self.device)
        self.local_batch = self.settings.local_batch
        self.local_epoch = self.settings.local_epoch
        self.optimizer = self.settings.optimizer
        self.lr = self.settings.lr
        self.momentum = self.settings.momentum
        self.weight_decay = self.settings.weight_decay
        #
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(indexes))

    def train_val_test(self, dataset, indexes):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # Split indexes for train, validation, and test (80, 10, 10)
        indexes_train = indexes[:int(0.8*len(indexes))]
        indexes_val = indexes[int(0.8*len(indexes)):int(0.9*len(indexes))]
        indexes_test = indexes[int(0.9*len(indexes)):]

        trainloader = DataLoader(DatasetSplit(dataset, indexes_train),
                                 batch_size=self.local_batch, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, indexes_val),
                                 batch_size=self.local_batch, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, indexes_test),
                                batch_size=self.local_batch, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, client):
        """
        Perform an iteration of local update.
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                        momentum=self.momentum)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        for iter in range(self.local_epoch):
            batch_loss = []
            for batch_index, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # Reset
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if batch_index == len(self.trainloader) - 1:
                    print(f'| Global Round : {global_round} | Client index : {client} | Local Epoch : {iter} | ' +  \
                          f'\tLoss: {loss.item():.3f}')
               
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ 
        Returns the inference accuracy and loss on the test-set.
        """
        # model in eval mode.
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct/total
        return acc, loss


