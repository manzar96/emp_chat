import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.transformers.modules_parlai import \
    TransformerEncodeDecoderVaswani, TransformerEncoder, TransformerDecoder


class TransformerVaswani_DoubleHead(TransformerEncodeDecoderVaswani):
    """
    This is a model with 2 heads.
    An LM head + a classification head
    """

    def __init__(self, opt, dictionary, embedding_size, embedding_weights=None,
                 num_classes=3, pad_idx=None, start_idx=None,
                 end_idx=None, device='cpu'):
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

        self.clf_layer = nn.Linear(opt.ffn_size, num_classes)

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output_lm = F.linear(tensor, self.embedding.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        # output[:, :, self.start_idx] = neginf(output.dtype)
        # TODO: maybe put average here!!
        output_clf = self.clf_layer(tensor[:,-1,:])
        return output_lm, output_clf

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
        logits_lm, logits_clf = self.output(latent)
        _, preds_lm = logits_lm.max(dim=2)
        _, preds_clf = logits_clf.max(dim=-1)
        return logits_lm, preds_lm, logits_clf, preds_clf

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
        # self.longest_label = max(self.longest_label, ys.size(1))
        # TODO: longest_label how to get rid of it?
        self.longest_label = ys.size(1)

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(
            *xs)
        # use teacher forcing
        scores_lm, preds_lm, scores_clf, preds_clf = self.decode_forced(encoder_states, ys)
        return scores_lm, preds_lm, scores_clf, preds_clf, encoder_states