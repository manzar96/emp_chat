# those modules are all taken from ParlAI
import numpy as np
import numbers
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Dict, Tuple, Optional
from core.modules.treesearch import GreedySearch, BeamSearch, TopKSampling, \
    NucleusSampling, DelayedBeamSearch
from torch.nn.parameter import Parameter

LAYER_NORM_EPS = 1e-5


def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    return -1e20


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm.
    """
    is_cpu = tensor.device == 'cpu' or tensor.device.type == 'cpu'
    return norm_layer(tensor)


def gelu(tensor):
    """
    Compute gelu function.

    c.f. https://arxiv.org/abs/1606.08415
    """
    return 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))


def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


class TransformerEncodeDecoderVaswani(nn.Module):
    def __init__(self, opt, dictionary, embedding_size, embedding_weights=None,
                 pad_idx=None, start_idx=None, end_idx=None, device='cpu'):
        super(TransformerEncodeDecoderVaswani, self).__init__()
        self.pad_idx = pad_idx
        self.end_idx = end_idx
        self.start_idx = start_idx
        self.device = device

        if embedding_weights is None and embedding_size is None:
            assert IOError, 'Provide pretrained embeddings or an embedding size'

        if embedding_weights is not None:
            assert not self.pad_idx, "pad_idx is None"
            print("Embeddings init with pretrained!")
            self.embedding = nn.Embedding(len(dictionary), embedding_size,
                                     padding_idx=self.pad_idx)
            self.embedding.weight = nn.Parameter(torch.from_numpy(
                embedding_weights),
                                            requires_grad=opt.learn_embeddings)
        else:
            self.embedding = nn.Embedding(len(dictionary), embedding_size,
                                     padding_idx=self.pad_idx)
            print("Embeddings init with normal distr!")
            nn.init.normal_(self.embedding.weight, 0, embedding_size ** -0.5)
            self.embedding.weight.requires_grad = opt.learn_embeddings

        # TODO: fix embeddings if its None!!

        self.encoder = TransformerEncoder(
            n_heads=opt.n_heads,
            n_layers=opt.n_layers,
            embedding_size=embedding_size,
            ffn_size=opt.ffn_size,
            vocabulary_size=len(dictionary),
            embedding=self.embedding,
            dropout=opt.dropout,
            attention_dropout=opt.attention_dropout,
            relu_dropout=opt.relu_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt.learn_positional_embeddings,
            embeddings_scale=opt.embeddings_scale,
            reduction_type=None,
            n_positions=opt.n_positions,
            n_segments=opt.n_segments,
            activation=opt.activation,
            variant=opt.variant,
            output_scaling=opt.output_scaling)

        self.decoder = TransformerDecoder(
            n_heads=opt.n_heads,
            n_layers=opt.n_layers,
            embedding_size=embedding_size,
            ffn_size=opt.ffn_size,
            vocabulary_size=len(dictionary),
            embedding=self.embedding,
            dropout=opt.dropout,
            attention_dropout=opt.attention_dropout,
            relu_dropout=opt.relu_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt.learn_positional_embeddings,
            embeddings_scale=opt.embeddings_scale,
            n_positions=opt.n_positions,
            activation=opt.activation,
            variant=opt.variant,
            n_segments=opt.n_segments)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embedding.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        #output[:, :, self.start_idx] = neginf(output.dtype)
        return output
    
    def decode_forced(self, encoder_states, targets):
        """
        Decode with a fixed, true sequence, computing loss.

        Useful for training, or ranking fixed candidates.

        :param targets:
            the prediction targets. Contains both the start and end tokens.

        :type targets:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, targets, vocab], LongTensor[bsz, targets])
        """
        bsz = targets.size(0)
        seqlen = targets.size(1)
        inputs = targets.narrow(1, 0, seqlen - 1)
        start_idxs = torch.LongTensor([self.start_idx]).expand(bsz, 1)
        start_idxs = start_idxs.to(self.device)
        inputs = torch.cat([start_idxs, inputs], dim=1)
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        #self.longest_label = max(self.longest_label, ys.size(1))
        # TODO: longest_label how to get rid of it?
        self.longest_label = ys.size(1)

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)
        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states


    def generate(self,*inputs, beam, max_ts,options):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        bsz = inputs[0].shape[0]
        encoder_states = self.encoder(*inputs)
        beams = [self.treesearch_factory() for i in range(bsz)]

    def _treesearch_factory(self, opt):
        method = opt.method
        beam_size = opt.beam_size
        if method == 'greedy':
            return GreedySearch(
                beam_size,
                min_length=0,
                block_ngram=opt.beam_block_ngram,
                context_block_ngram=opt.beam_context_block_ngram,
                length_penalty=opt.beam_length_penalty,
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=self.device,
            )
        elif method == 'beam':
            return BeamSearch(
                beam_size,
                min_length=opt.beam_min_length,
                block_ngram=opt.beam_block_ngram,
                context_block_ngram=opt.beam_context_block_ngram,
                length_penalty=opt.beam_length_penalty,
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=self.device,
            )
        elif method == 'delayedbeam':
            return DelayedBeamSearch(
                opt.topk,
                opt.beam_delay,
                beam_size,
                min_length=opt.beam_min_length,
                block_ngram=opt.beam_block_ngram,
                context_block_ngram=opt.beam_context_block_ngram,
                length_penalty=opt.beam_length_penalty,
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=self.device,
            )
        elif method == 'topk':
            return TopKSampling(
                opt.topk,
                beam_size,
                min_length=opt.beam_min_length,
                block_ngram=opt.beam_block_ngram,
                context_block_ngram=opt.beam_context_block_ngram,
                length_penalty=opt.beam_length_penalty,
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=self.device,
            )
        elif method == 'nucleus':
            return NucleusSampling(
                opt.topp,
                beam_size,
                min_length=opt.beam_min_length,
                block_ngram=opt.beam_block_ngram,
                context_block_ngram=opt.beam_context_block_ngram,
                length_penalty=opt.beam_length_penalty,
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=self.device,
            )
        else:
            raise ValueError(f"Can't use inference method {method}")


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = dropout
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions #used for positional embeddings!
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm' or self.variant == 'prelayernorm':
            self.norm_embeddings = nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                )
            )
        self.output_scaling = output_scaling

    def forward(self, input, positions=None, segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param BoolTensor[batch,seqlen] mask:
            The attention mask; 1 means attend, 0 means ignore.
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warnings.warn(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    # def _apply_model_parallel(self, tensor, mask):
    #     """
    #     Pipeline application of model parallelism.
    #     """
    #     chunks = PipelineHelper.split((tensor, mask))
    #     work_items = PipelineHelper.schedule_work_items(self.layers, chunks)
    #
    #     for chunk_idx, layer_nos, next_device in work_items:
    #         s_tensor, s_mask = chunks[chunk_idx]
    #         for layer_no in layer_nos:
    #             s_tensor = self.layers[layer_no](s_tensor, s_mask)
    #         chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_mask), next_device)
    #
    #     tensor_out, mask_out = PipelineHelper.join(chunks)
    #     return tensor_out


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        """
        Forward pass.
        """

        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm1)
        attended_tensor, _ = self.attention(tensor, mask=mask)
        tensor = residual + self.dropout(attended_tensor)
        if self.variant == 'aiayn' or self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm1)
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm2)
        tensor = residual + self.dropout(self.ffn(tensor))
        if self.variant == 'aiayn' or self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant

        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if self.variant == 'xlm' or self.variant == 'prelayernorm':
            self.norm_embeddings = nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )
            )

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warnings.warn(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    x=tensor,
                    encoder_output=encoder_output,
                    encoder_mask=encoder_mask,
                    incr_state=incr_state.get(idx),
                )

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor, new_incr_state

    # def _apply_model_parallel(self, tensor, encoder_output, encoder_mask, incr_state):
    #     """
    #     Pipeline application of model parallelism.
    #     """
    #     chunks = PipelineHelper.split(
    #         (tensor, encoder_output, encoder_mask, incr_state)
    #     )
    #     work_items = PipelineHelper.schedule_work_items(self.layers, chunks)
    #
    #     new_incr_state = [{} for _ in chunks]
    #
    #     for chunk_idx, layer_nos, next_device in work_items:
    #         s_tensor, s_enc_out, s_enc_mask, s_incr_state = chunks[chunk_idx]
    #         for layer_no in layer_nos:
    #             s_tensor, new_incr_state[chunk_idx][layer_no] = self.layers[layer_no](
    #                 x=s_tensor,
    #                 encoder_output=s_enc_out,
    #                 encoder_mask=s_enc_mask,
    #                 incr_state=s_incr_state.get(layer_no),
    #             )
    #         chunks[chunk_idx] = PipelineHelper.chunk_to(
    #             (s_tensor, s_enc_out, s_enc_mask, s_incr_state), next_device
    #         )
    #
    #     tensor_out = PipelineHelper.join([c[0] for c in chunks])
    #     new_incr_state = PipelineHelper.join(new_incr_state)
    #
    #     return tensor_out, new_incr_state


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask, incr_state=None):
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm1)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
        )
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm':
            x = _normalize(x, self.norm1)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm2)
        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm':
            x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm3)
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm':
            x = _normalize(x, self.norm3)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }
        return x, new_incr_state

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }


class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(self, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x


class BasicAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self, dim=1, attn='cosine', residual=False, get_weights=True):
        super().__init__()
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None, values=None):
        """
        Compute attention.

        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)

        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        bsz = xs.size(0)
        y_len = ys.size(1)
        x_len = xs.size(1)
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).view(bsz, 1, y_len)
            attn_mask = attn_mask.repeat(1, x_len, 1)
            l1.masked_fill(attn_mask, neginf(l1.dtype))
        l2 = F.softmax(l1, dim=self.dim, dtype=torch.float).type_as(l1)
        if values is None:
            values = ys
        lhs_emb = torch.bmm(l2, values)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb.squeeze(self.dim - 1)


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(  # type: ignore
        # TODO: remove type ignore with pytorch 1.5:
        # https://github.com/pytorch/pytorch/pull/31057
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, 1] (decoder self-attn with incr_state caching)
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :return: (final attended tensor, new incremental state)
        """

        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        # Prepend incremental states. For each of the key, value, and mask, see if
        # a previous incremental state exists, and if so, reshape it to match the shape
        # of the new state. Concatenate the previous and new states to match what the
        # full state would have been if we had not cached. (If we are using static_kv,
        # these three states are unchanging, so just re-use the cached states.)
        if incr_state is None:
            incr_state = {}
        if 'prev_key' in incr_state:
            prev_key = incr_state['prev_key'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in incr_state:
            prev_value = incr_state['prev_value'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)
        if 'prev_mask' in incr_state:
            if static_kv:
                mask = incr_state['prev_mask']
            else:
                mask = torch.cat([incr_state['prev_mask'], mask], dim=2)
                # Prepend along the key_len dimension (analogous to
                # incr_state['prev_key'])

        # Save new incremental states. We reshape to allow for reordering along batch
        # dimension.
        new_incr_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask,
        }

        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, full_key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, full_key_len)
            .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1, dtype=torch.float).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out, new_incr_state

    def reorder_incremental_state(
        self, incremental_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reorder the input incremental-state tensors.
        """
        return {
            key: torch.index_select(val, 0, inds.to(val.device)).contiguous()
            for key, val in incremental_state.items()
        }

