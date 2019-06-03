from options import opt
import logging
import random
import numpy as np
import torch
from my_utils import makedir_and_clear, Alphabet, MyDataset, save, load
from rdoc_preprocess import load_data, build_alphabet, prepare_instance, my_collate, evaluate
from models import BertForSequenceClassification_rdoc
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os


if __name__ == "__main__":

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)

    if opt.whattodo == 'train':

        makedir_and_clear(opt.save)

        documents, alphabet_category = load_data(opt.train)
        # no test set, random split 20%
        train_size = int(len(documents)*0.8)
        random.shuffle(documents)
        train_documents = documents[:train_size]
        test_documents = documents[train_size:]
        logging.info("{} documents in train set".format(len(train_documents)))
        logging.info("{} documents in test set".format(len(test_documents)))

        alphabet_label = Alphabet('label', True)
        build_alphabet(train_documents, alphabet_label)
        logging.info("label: {}".format(alphabet_label.instances))

        train_instances = prepare_instance(train_documents, alphabet_label, alphabet_category)
        logging.info("train instance number: {}".format(len(train_instances)))

        if opt.gpu >= 0 and torch.cuda.is_available():
            device = torch.device("cuda", opt.gpu)
        else:
            device = torch.device("cpu")
        logging.info("use device {}".format(device))

        model = BertForSequenceClassification_rdoc.from_pretrained(opt.bert_dir, num_labels=alphabet_label.size(),
                                                                   num_category=alphabet_category.size())
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)

        train_loader = DataLoader(MyDataset(train_instances), opt.batch_size, shuffle=True, collate_fn=my_collate)

        logging.info("start training ...")

        best_test = -10
        bad_counter = 0
        for idx in range(opt.iter):
            epoch_start = time.time()

            model.train()

            train_iter = iter(train_loader)
            num_iter = len(train_loader)

            sum_loss = 0
            correct, total = 0, 0

            for i in range(num_iter):
                tokens, labels, mask, sent_type, cate = next(train_iter)

                logits = model.forward(cate, tokens, sent_type, mask)

                loss, total_this_batch, correct_this_batch = model.loss(logits, labels)

                sum_loss += loss.item()

                loss.backward()
                optimizer.step()
                model.zero_grad()

                total += total_this_batch
                correct += correct_this_batch

            epoch_finish = time.time()
            accuracy = 100.0 * correct / total
            logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
                idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

            precision, recall, f_measure = evaluate(test_documents, model, alphabet_label, alphabet_category, None)
            logging.info("Test p=%.4f, r=%.4f, f1=%.4f" % (precision, recall, f_measure))
            if f_measure > best_test:
                logging.info("Exceed previous best performance on test: %.4f" % (f_measure))
                best_test = f_measure
                bad_counter = 0

                torch.save(model.state_dict(), os.path.join(opt.save, 'model.pth'))
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                logging.info('Early Stop!')
                break

        # save some information
        save(alphabet_label, os.path.join(opt.save, 'alphabet_label.pkl'))
        save(alphabet_category, os.path.join(opt.save, 'alphabet_category.pkl'))

    elif opt.whattodo == 'test':

        makedir_and_clear(opt.predict)

        test_documents, _ = load_data(opt.test)

        alphabet_label = Alphabet('label', True)
        load(alphabet_label, os.path.join(opt.save, 'alphabet_label.pkl'))
        alphabet_category = Alphabet('category', True)
        load(alphabet_category, os.path.join(opt.save, 'alphabet_category.pkl'))

        model = BertForSequenceClassification_rdoc.from_pretrained(opt.bert_dir, num_labels=alphabet_label.size(),
                                                                   num_category=alphabet_category.size())
        if opt.gpu >= 0 and torch.cuda.is_available():
            model.load_state_dict(
                torch.load(os.path.join(opt.save, 'model.pth'), map_location='cuda:{}'.format(opt.gpu)))
        else:
            model.load_state_dict(torch.load(os.path.join(opt.save, 'model.pth'), map_location='cpu'))

        logging.info("start test ...")
        evaluate(test_documents, model, alphabet_label, alphabet_category, opt.predict)


    logging.info("end ......")