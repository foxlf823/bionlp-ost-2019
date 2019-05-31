
import os
import codecs
import nltk
nlp_tool = nltk.data.load('tokenizers/punkt/spanish.pickle')
from pytorch_pretrained_bert import BertTokenizer
from options import opt
wp_tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
from my_utils import my_tokenize, pad_sequence
import logging
from torch.utils.data import DataLoader

class Document:
    def __init__(self):
        self.entities = []
        self.sentences = []
        self.name = ''
        self.text = ''

class Sentence:
    def __init__(self):
        self.tokens = []
        self.start = -1
        self.end = -1

class Entity:
    def __init__(self):
        self.id = ''
        self.type = ''
        self.start = -1
        self.end = -1
        self.text = ''
        self.sent_idx = -1
        self.tf_start = -1
        self.tf_end = -1

    def create(self, id, type, start, end, text, sent_idx, tf_start, tf_end):
        self.id = id
        self.type = type
        self.start = start
        self.end = end
        self.text = text
        self.sent_idx = sent_idx
        self.tf_start = tf_start
        self.tf_end = tf_end

    def append(self, start, end, text, tf_end):

        whitespacetoAdd = start - self.end
        for _ in range(whitespacetoAdd):
            self.text += " "
        self.text += text

        self.end = end
        self.tf_end = tf_end

    def getlength(self):
        return self.end-self.start

    def equals(self, other):
        if self.type == other.type and self.start == other.start and self.end == other.end:
            return True
        else:
            return False




def load_data(data_dir, bert_dir):

    doc_num = 0
    sent_num = 0
    entity_num = 0
    max_sent_length = 0
    min_sent_length = 9999
    total_sent_length = 0

    documents = []
    for input_file_name in os.listdir(data_dir):
        if input_file_name.find(".txt") != -1:
            document = Document()
            document.name = input_file_name


            ann_file_name = input_file_name.replace(".txt", '.ann')
            if os.path.isfile(os.path.join(data_dir,ann_file_name)):
                with codecs.open(os.path.join(data_dir, ann_file_name), 'r', 'UTF-8') as fp:

                    for line in fp:
                        line = line.strip()
                        if line == '':
                            continue
                        entity = {}
                        columns = line.split('\t')
                        entity['id'] = columns[0]
                        columns_1 = columns[1].split(" ")
                        entity['type'] = columns_1[0]
                        entity['start'] = int(columns_1[1])
                        entity['end'] = int(columns_1[2])
                        entity['text'] = columns[2]

                        document.entities.append(entity)
                        entity_num += 1


            with codecs.open(os.path.join(data_dir,input_file_name), 'r', 'UTF-8') as fp:
                document.text = fp.read()

            all_sents_inds = []
            generator = nlp_tool.span_tokenize(document.text)
            for t in generator:
                all_sents_inds.append(t)


            for ind in range(len(all_sents_inds)):
                sentence = Sentence()
                sentence.start = all_sents_inds[ind][0]
                sentence.end = all_sents_inds[ind][1]

                offset = 0
                sentence_txt = document.text[sentence.start:sentence.end]
                for token_txt in my_tokenize(sentence_txt):
                    token = {}
                    offset = sentence_txt.find(token_txt, offset)
                    if offset == -1:
                        raise RuntimeError("can't find {} in '{}'".format(token_txt, sentence_txt))

                    token['text'] = token_txt
                    token['start'] = sentence.start + offset
                    token['end'] = sentence.start + offset + len(token_txt)
                    token['wp'] = wp_tokenizer.tokenize(token_txt)
                    if len(document.entities) != 0:
                        token['label'] = getLabel_BIO(token['start'], token['end'], document.entities)

                    sentence.tokens.append(token)
                    offset += len(token_txt)

                document.sentences.append(sentence)
                sent_num += 1
                total_sent_length += len(sentence.tokens)
                if len(sentence.tokens) > max_sent_length:
                    max_sent_length = len(sentence.tokens)
                if len(sentence.tokens) < min_sent_length:
                    min_sent_length = len(sentence.tokens)




            documents.append(document)
            doc_num += 1

    logging.info("{} statistics".format(data_dir))
    logging.info("doc number {}, sent number {}, entity number {}".format(doc_num, sent_num, entity_num))
    logging.info("avg sent length {}, max sent length {}, min sent length {}".format(total_sent_length//sent_num,
                                                                                     max_sent_length, min_sent_length))
    return documents



def getLabel_BIO(start, end, entities):

    match = ""
    for index, entity in enumerate(entities):
        if start == entity['start'] and end == entity['end'] : # B
            match = "B"
            break
        elif start == entity['start'] and end != entity['end'] : # B
            match = "B"
            break
        elif start != entity['start'] and end == entity['end'] : # I
            match = "I"
            break
        elif start > entity['start'] and end < entity['end']:  # I
            match = "I"
            break

    if match != "":
        return match+"-"+entity['type']
    else:
        return "O"

def build_alphabet(documents, label_alphabet):
    for document in documents:
        for sentence in document.sentences:
            for token in sentence.tokens:
                label_alphabet.add(token['label'])

    label_alphabet.add("X")


# one sentence one instance
def prepare_instance_for_one_doc(document, alphabet_label):
    instances = []

    for sentence in document.sentences:

        tokens = []
        labels = []

        for token in sentence.tokens:
            word_pieces = token['wp']
            word_pieces_labels = ["X"] * len(word_pieces)
            word_pieces_labels[0] = token['label']

            tokens.extend(word_pieces)
            labels.extend(word_pieces_labels)

        if len(tokens) > opt.len_max_seq-2: # for [CLS] and [SEP]
            tokens = tokens[:opt.len_max_seq-2]
            labels = labels[:opt.len_max_seq-2]

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        tokens = wp_tokenizer.convert_tokens_to_ids(tokens)
        labels.insert(0, 'O')
        labels.append('O')
        labels = [alphabet_label.get_index(wpl) for wpl in labels]

        instance = {'tokens': tokens, 'labels': labels}

        instances.append(instance)

    return instances


def prepare_instance(documents, alphabet_label):
    instances = []

    for document in documents:

        instances_one_doc = prepare_instance_for_one_doc(document, alphabet_label)

        instances.extend(instances_one_doc)

    return instances


import torch
def my_collate(x):
    tokens = [x_['tokens'] for x_ in x]
    labels = [x_['labels'] for x_ in x]
    lengths = [len(x_['tokens']) for x_ in x]
    max_len = max(lengths)
    mask = [[1] * length for length in lengths]
    sent_type = [[0]* length for length in lengths]

    tokens = pad_sequence(tokens, max_len)
    labels = pad_sequence(labels, max_len)
    mask = pad_sequence(mask, max_len)
    sent_type = pad_sequence(sent_type, max_len)

    if opt.gpu >= 0 and torch.cuda.is_available():
        tokens = tokens.cuda(opt.gpu)
        labels = labels.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        sent_type = sent_type.cuda(opt.gpu)

    return tokens, labels, mask, sent_type

from my_utils import MyDataset
def evaluate(documents, model, alphabet_label, dump_dir):

    ct_predicted = 0
    ct_gold = 0
    ct_correct = 0

    for document in documents:
        instances = prepare_instance_for_one_doc(document, alphabet_label)

        data_loader = DataLoader(MyDataset(instances), opt.batch_size, shuffle=False, collate_fn=my_collate)

        pred_entities = []

        with torch.no_grad():
            model.eval()

            data_iter = iter(data_loader)
            num_iter = len(data_loader)
            sent_start = 0
            entity_id = 1

            for i in range(num_iter):
                tokens, labels, mask, sent_type = next(data_iter)

                logits = model.forward(tokens, sent_type, mask)

                actual_batch_size = tokens.size(0)

                for batch_idx in range(actual_batch_size):

                    sent_logits = logits[batch_idx]
                    sent_mask = mask[batch_idx]

                    _, indices = torch.max(sent_logits, 1)

                    actual_indices = indices[sent_mask == 1]

                    actual_indices = actual_indices[1:-1] # remove [CLS] and [SEP]

                    pred_labels = [alphabet_label.get_instance(ind.item()) for ind in actual_indices]

                    sentence = document.sentences[sent_start + batch_idx]

                    sent_entities = translateLabelintoEntities(pred_labels, sentence, entity_id, sent_start + batch_idx)
                    entity_id += len(sent_entities)

                    pred_entities.extend(sent_entities)

                sent_start += actual_batch_size

        if len(document.entities) != 0:
            p1, p2, p3 = count_tp(document.entities, pred_entities)
        else:
            p1, p2, p3 = 0, 0, 0

        # dump results
        if dump_dir:
            dump_results(document, pred_entities, dump_dir)


        ct_gold += p1
        ct_predicted += p2
        ct_correct += p3

    if ct_gold == 0 or ct_predicted == 0:
        precision = 0
        recall = 0
    else:
        precision = ct_correct * 1.0 / ct_predicted
        recall = ct_correct * 1.0 / ct_gold

    if precision+recall == 0:
        f_measure = 0
    else:
        f_measure = 2*precision*recall/(precision+recall)

    return precision, recall, f_measure




def translateLabelintoEntities(pred_labels, sentence, entity_id, sent_id):

    results = []
    labelSequence = []
    pred_label_idx = 0

    for token_idx, token in enumerate(sentence.tokens):

        # pred_labels may be short than sentence.tokens, since len_max_seq
        # so we ignore the token that don't have pred_labels
        if pred_label_idx >= len(pred_labels):
            break

        # look at the label of the first wp to determine the label of that token
        pred_label = pred_labels[pred_label_idx]
        labelSequence.append(pred_label)

        if pred_label[0] == 'B':
            entity = Entity()
            entity.create(str(entity_id), pred_label[2:], token['start'], token['end'], token['text'], sent_id, token_idx, token_idx)
            results.append(entity)
            entity_id += 1

        elif pred_label[0] == 'I':
            if checkWrongState(labelSequence):
                entity = results[-1]
                entity.append(token['start'], token['end'], token['text'], token_idx)

        pred_label_idx += len(token['wp'])

    return results



def checkWrongState(labelSequence):
    positionNew = -1
    positionOther = -1
    currentLabel = labelSequence[-1]
    assert currentLabel[0] == 'I'

    for j in range(len(labelSequence)-1)[::-1]:
        if positionNew == -1 and currentLabel[2:] == labelSequence[j][2:] and labelSequence[j][0] == 'B' :
            positionNew = j
        elif positionOther == -1 and (currentLabel[2:] != labelSequence[j][2:] or labelSequence[j][0] != 'I'):
            positionOther = j

        if positionOther != -1 and positionNew != -1:
            break

    if positionNew == -1:
        return False
    elif positionOther < positionNew:
        return True
    else:
        return False


def count_tp(gold_entities, pred_entities):

    ct_gold = len(gold_entities)
    ct_predict = len(pred_entities)
    ct_correct = 0

    for predict_entity in pred_entities:

        for gold_entity in gold_entities:

            if predict_entity.type==gold_entity['type'] and predict_entity.start==gold_entity['start'] and \
                    predict_entity.end==gold_entity['end']:
                ct_correct += 1
                break

    return ct_gold, ct_predict, ct_correct


def dump_results(document, pred_entities, dump_dir):

    with codecs.open(os.path.join(dump_dir, document.name), 'w', 'UTF-8') as fp:

        fp.write(document.text)

    with codecs.open(os.path.join(dump_dir, document.name.replace('.txt', '.ann')), 'w', 'UTF-8') as fp:

        for entity in pred_entities:
            ss = "T" + str(entity.id) + "\t" + entity.type + " " + str(entity.start) + " " + str(entity.end)\
                 + "\t" + entity.text + "\n"
            fp.write(ss)


