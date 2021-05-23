import numpy as np
from tqdm import tqdm
def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True
                                       )

        input_ids = inputs["input_ids"]
        # print(tokenizer.decode(input_ids))
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.query, instance.doc

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def _convert_to_transformer_inputs_single(doc, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True
                                       )

        input_ids = inputs["input_ids"]
        # print(tokenizer.decode(input_ids))
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
    doc, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays_single(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    
    for _, instance in tqdm(df[columns].iterrows()):
        d = instance.doc

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs_single(d, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]
