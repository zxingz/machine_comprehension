# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import six
import tensorflow as tf
from uuid import uuid4
import sys
from ConfigParser import ConfigParser
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec

appDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # Declaring Application Directory

sys.path.append(appDir)  # Adding code directory path for standalone execution
config = ConfigParser()
config.read(os.path.join(appDir, 'config.cfg'))
modelDir = os.path.realpath(config.get('BERT', 'dest_model_dir'))
if __name__ == '__main__':
    modelDir = os.path.join(appDir, 'model')

from BERT import modeling
from BERT import optimization
from BERT import tokenization

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


class SquadExample(object):
    """A single training/test example for simple sequence classification.
       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_QA(qa, paragraphs):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []

    for paragraph in paragraphs:
        paragraph_text = paragraph['text']
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        qas_id = paragraph["id"]
        question_text = qa
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)
    return examples


def input_fn_builder(features, seq_length, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_start_positions = []
    all_end_positions = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_start_positions.append(feature.start_position)
        all_end_positions.append(feature.end_position)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        feature_map = {
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        }

        d = tf.data.Dataset.from_tensor_slices(feature_map)
        d = d.repeat()
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        predictions = {
            "unique_ids": unique_ids,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        output_spec = EstimatorSpec(mode=mode, predictions=predictions, scaffold=scaffold_fn)
        return output_spec

    return model_fn


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (unique_id))
            tf.logging.info("example_index: %s" % (example_index))
            tf.logging.info("doc_span_index: %s" % (doc_span_index))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("token_to_orig_map: %s" % " ".join(
                ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
            tf.logging.info("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
            ]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

            start_position = None
            end_position = None
            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            features.append(feature)
            unique_id += 1

    return features



def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def write_QA(all_examples, all_features, all_results, n_best_size,
             max_answer_length, do_lower_case):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing nbest ")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="", start_logit=null_start_logit,
                    end_logit=null_end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        nbest_json = []
        if best_non_null_entry is not None:

            probs = _compute_softmax(total_scores)

            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > 0.0:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

            all_nbest_json[example.qas_id] = nbest_json
        else:
            output = collections.OrderedDict()
            output["text"] = ''
            output["probability"] = 0
            output["start_logit"] = 0
            output["end_logit"] = 0
            nbest_json.append(output)

    return all_nbest_json


def process(questions, context):
    # TODO Replace all abbreviation code

    bert_config = modeling.BertConfig.from_json_file(os.path.join(modelDir, 'bert_config.json'))  # Loading bert config
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(modelDir, 'vocab.txt'),
                                           do_lower_case=True)  # Loading tokenizer

    eval_examples = read_QA(questions, context)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=512,
        doc_stride=256,
        max_query_length=128)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=os.path.join(modelDir, 'bert_model.ckpt'),
        use_one_hot_embeddings=False)

    run_config = RunConfig(model_dir=modelDir, save_checkpoints_steps=1000)
    estimator = Estimator(model_fn=model_fn, config=run_config, params={'batch_size': 14})

    predict_input_fn = input_fn_builder(
        features=eval_features,
        seq_length=512,
        drop_remainder=True)

    all_results = []
    counter = 0
    for result in estimator.predict(predict_input_fn, yield_single_examples=True):
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
        counter += 1
        if len(eval_features) == counter: break

    all_nbest_json = write_QA(eval_examples, eval_features, all_results,
                                                          2, 128, False)
    return all_nbest_json


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    bertQuestion = 'who is chief operating officer of enercare ?'
    """
    bertContext = [{"id": 1,
                    "text": '''Upon his retirement, Jenine Krause, Chief Operating Officer, Home Services has been
appointed President and Chief Executive Officer of Enercare.'''},
                   {"id": 2,
                    "text": '''Upon his retirement, Jenine Krause, Chief Operating Officer, Home Services has been
    appointed President and Chief Executive Officer of Enercare.'''},
                   ]
    """
    bertContext = [{'text': u'Date: Ticker Symbol: 07-Feb-2019   ALS.TSX   Civic Address: Altius Minerals Corporation Suite 202, 66 Kenmount Road St. Johns, NL, A1B 3V7, CANADA Website: www.altiusminerals.com Fax: 709.576.3441 e-mail: info@altiusminerals.com Mailing Address: Altius Minerals Corporation P.O. Box 8263 Stn A St. Johns, NL, A1B 3N4, CANADA Maritime Electric, Newfoundland Power and Fortis Alberta and then CEO roles at subsidiaries Fortis Properties and Newfoundland Power. Brian Dalton, President and CEO of Altius, commented, Altius has now taken the creation of a royalty business based around long-life renewable energy strong execution and strategic growth, said John Billingsley, Chairman and CEO of Tri Global Energy. 2018 also was one of our best years for completing and initiating renewable energy development projects. This transaction with Altius will accelerate our development and allow us to extend our involvement in these projects financing products like the renewable royalty investment provided by Altius. More information on the transaction and ARR, including an investor presentation, can be found at http://www.altiusminerals.com/. About Altius Altius directly and indirectly holds diversified royalties and streams which generate royalties covering a wide spectrum of mineral commodities and jurisdictions. Altius also holds a large portfolio of exploration stage projects which it has generated that results in newly created royalties and equity and minority interests. Altius has 42,851,726 common shares issued and outstanding that are listed on Canadas Toronto', 'id': 0}, {'text': u'Date: Ticker Symbol: 07-Feb-2019   ALS.TSX   Civic Address: Altius Minerals Corporation Suite 202, 66 Kenmount Road St. Johns, NL, A1B 3V7, CANADA Website: www.altiusminerals.com Fax: 709.576.3441 e-mail: info@altiusminerals.com Mailing Address: Altius Minerals Corporation P.O. Box 8263 Stn A St. Johns, NL, A1B 3N4, CANADA   Altius Announces First Renewable Energy Royalty Transaction - US$30MM transaction with leading U.S. wind energy developer - St. Johns - Altius Minerals Corporation (Altius) (ALS:TSX, ATUSF: OTCQX) is pleased to announce that its recently formed subsidiary, Altius Renewable Royalties Corp. (ARR), has entered into a transaction with Tri Global Energy projects added in the future, to this new royalty investment structure with Altius, until a minimum total royalty portfolio valuation threshold is achieved. The currently power price changes. Funding for the TGE transaction will be provided from Altius available cash on hand. Depending upon the scale of the overall opportunity set that develops for ARR, Altius may also consider adding additional future investment partners as either direct participants or through the creation of limited partnership structures. Altius Renewable Royalties As part of its founding, ARR has acquired a private company, Great and elsewhere. This management team, led by Frank Getman (President and CEO), has a long and successful track record of developing and operating small and large-scale the Board of Directors, joining Mr. Getman and parent level appointees of Altius Minerals Corporation that include its CEO and CFO. Earl retired at the end of 2017 from Fortis Inc. which is a major international electrical', 'id': 1}, {'text': u'Voiseys Bay The royalty on production of nickel, copper, cobalt and other minerals from the Voiseys Bay mine in Newfoundland and Labrador, Canada is directly owned 90% owner. The remaining 10% interest in LNRLP is owned by a subsidiary of Altius Minerals Corporation (Altius), a non-controlling interest. On September 13, 2018, LNRLP entered into an agreement with', 'id': 2}, {'text': u'Voiseys Bay The royalty on production of nickel, copper, cobalt and other minerals from the Voiseys Bay mine in Newfoundland and Labrador, Canada is directly owned 90% owner. The remaining 10% interest in LNRLP is owned by a subsidiary of Altius Minerals Corporation (Altius), a non-controlling interest. On September 13, 2018, LNRLP entered into an agreement with', 'id': 3}, {'text': u'Voiseys Bay The royalty on production of nickel, copper, cobalt and other minerals from the Voiseys Bay mine in Newfoundland and Labrador, Canada is directly owned 90% owner. The remaining 10% interest in LNRLP is owned by a subsidiary of Altius Minerals Corporation (Altius). On September 13, 2018, LNRLP entered into an agreement with Vale Canada Limited and certain', 'id': 4}]
    print(process(questions=bertQuestion, context=bertContext))
