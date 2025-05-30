import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from shared.utils import batched_index_select
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from shared.const import task_ner_labels

# import pyreft


class NERModel(nn.Module):
    def __init__(
        self,
        config,
        num_ner_labels,
        width_embedding_dim=150,
        max_span_length=8,
        mask_range=None,
        args=None,
        one_hot_label=None,
    ):
        super(NERModel, self).__init__()
        self.args = args
        args.logger.info(f"num_ner_labels: {num_ner_labels}")
        self.num_ner_labels = num_ner_labels
        self.one_hot_label = one_hot_label
        if args.vae:
            conditional_labels_dim = 0
        elif args.corpus_one_hot:
            conditional_labels_dim = num_ner_labels + len(args.task_list)
        elif (
            args.general_one_hot
            or args.lookup_label_token_onehot == "low"
            or args.lookup_label_token_onehot == "random"
        ):
            if len(args.task_list) == 1:
                mask_range = [(0, num_ner_labels)]
            conditional_labels_dim = num_ner_labels + mask_range[0][1]
        elif (
            args.lookup_label_token_onehot == "separate"
            or args.lookup_label_token_onehot == "separate_learn"
        ):
            conditional_labels_dim = num_ner_labels
        elif args.lookup_label_token_onehot == "only_shared":
            conditional_labels_dim = mask_range[0][1]
        elif (
            args.label_token
            or args.shared_label_token
            or args.lookup_label_token
            or args.lookup_label_token_onehot == "high"
        ) and not args.simple:
            conditional_labels_dim = config.hidden_size
        elif args.condition_dynamic_mu:
            conditional_labels_dim = args.z_hidden_dim
        elif args.saved_label_embedding_dir is not None:
            saved_label_embedding = torch.load(args.saved_label_embedding_dir)
            conditional_labels_dim = saved_label_embedding.size(-1)
        else:
            conditional_labels_dim = num_ner_labels

        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)

        self.spans_embedding_layer = nn.Linear(
            config.hidden_size * 2 + width_embedding_dim,
            config.hidden_size,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.ner_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                config.hidden_size // 2, num_ner_labels
            ),  # head_hidden_dim # config.hidden_size
        )
        if args.lookup_label_token:
            self.label_embedding = nn.Embedding(num_ner_labels, config.hidden_size)
        elif (
            args.lookup_label_token_onehot == "low" or args.lookup_label_token_onehot == "separate"
        ):
            self.label_embedding = nn.Embedding.from_pretrained(
                one_hot_label, freeze=args.condition_freezing
            )
        elif args.lookup_label_token_onehot == "only_shared":
            self.label_embedding = nn.Embedding.from_pretrained(
                one_hot_label, freeze=args.condition_freezing
            )
        elif args.lookup_label_token_onehot == "high":
            one_hot_label_tuple = (
                one_hot_label,
                torch.zeros(
                    one_hot_label.size(0),
                    config.hidden_size - one_hot_label.size(1),
                ).to(one_hot_label.device),
            )
            self.label_embedding = nn.Embedding.from_pretrained(
                torch.cat(one_hot_label_tuple, dim=1),
                freeze=args.condition_freezing,
            )
        elif args.lookup_label_token_onehot == "random":
            self.label_embedding = nn.Embedding(num_ner_labels, conditional_labels_dim)
        elif args.saved_label_embedding_dir is not None:
            self.label_embedding = nn.Embedding.from_pretrained(
                saved_label_embedding, freeze=args.condition_freezing
            )

        if not args.simple:
            self.z_hidden_dim = args.z_hidden_dim
            reconst_dim = config.hidden_size
            cvae_input_dim = config.hidden_size

            if args.dynamic_mu == "random":
                self.dynamic_mu = nn.Embedding(num_ner_labels, self.z_hidden_dim)
            elif args.dynamic_mu == "fixed_bert":
                self.dynamic_mu = self.make_label_embedding()
                self.z_hidden_dim = self.dynamic_mu.size(-1)
            elif args.dynamic_mu == "bert_with_layer":
                self.dynamic_mu = self.make_label_embedding()
                self.dynamic_mu_layer = nn.Linear(self.dynamic_mu.size(-1), self.z_hidden_dim)
            elif args.dynamic_mu == "t5_with_layer" or self.args.condition_dynamic_mu:
                self.dynamic_mu = self.make_label_embedding(tinybert=False)
                self.dynamic_mu_layer = nn.Linear(self.dynamic_mu.size(-1), self.z_hidden_dim)

            self.span_layer = nn.Linear(
                cvae_input_dim + conditional_labels_dim,
                args.enc_hidden_dim,
            )
            self.mu = nn.Linear(args.enc_hidden_dim, self.z_hidden_dim)
            self.log_var = nn.Linear(
                args.enc_hidden_dim, self.z_hidden_dim
            )  # reconst_dim + conditional_labels_dim

            # for decoder
            self.dec_linear1 = nn.Linear(
                self.z_hidden_dim + conditional_labels_dim, self.z_hidden_dim
            )
            self.dec_linear2 = nn.Linear(self.z_hidden_dim, reconst_dim)

    def encode_span_embeddings(
        self,
        input_ids,
        spans,
        token_type_ids=None,
        attention_mask=None,
        corpus_info=None,
    ):
        if self.args.use_t5:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,  # two or more sentences input
                attention_mask=attention_mask,
            )

        sequence_output = outputs.last_hidden_state
        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(sequence_output, spans_start)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1
        )
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding  # , label_tensor

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.args.model_device)
        return mu + eps * std

    def decorde_span_embeddings(self, z):
        t = F.relu(self.dec_linear1(z))
        t = F.relu(self.dec_linear2(t))
        return t

    def cvae_loss_function(self, x, pred, mu, logvar):
        recon_loss = F.mse_loss(pred, x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2), dim=(0, 1))
        return recon_loss + self.args.beta * kld

    def dynamic_cvae_loss_function(self, x, pred, mu, logvar, spans_ner_label):
        recon_loss = F.mse_loss(pred, x)
        if self.args.dynamic_mu == "bert_with_layer" or self.args.dynamic_mu == "t5_with_layer":
            dynamic_mu = F.relu(self.dynamic_mu_layer(self.dynamic_mu))
            span_dynamic_mu = dynamic_mu[spans_ner_label]
        else:
            span_dynamic_mu = self.dynamic_mu[spans_ner_label]
        mu_diff = mu - span_dynamic_mu
        kld = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu_diff.pow(2) - logvar.exp(), dim=2),
            dim=(0, 1),
        )
        return recon_loss + self.args.beta * kld

    def make_label_embedding(self, tinybert=True):
        """make label embedding by encoding label text"""
        if tinybert:
            model_name = "huawei-noah/TinyBERT_General_4L_312D"
            label_encoder = AutoModel.from_pretrained(model_name)
            label_tokenizer = AutoTokenizer.from_pretrained(model_name)
            end_token = label_tokenizer.sep_token
            end_token_id = label_tokenizer.sep_token_id
        else:
            model_name = self.args.model
            label_encoder = T5EncoderModel.from_pretrained(model_name)
            label_tokenizer = AutoTokenizer.from_pretrained(model_name)
            end_token = label_tokenizer.eos_token
            end_token_id = label_tokenizer.eos_token_id
        if self.args.label_emb_input == "sent":
            label_text = (
                [
                    f"{label_tokenizer.cls_token} {label} in {corpus} {end_token}"
                    for corpus in self.args.task_list
                    for label in task_ner_labels[corpus]
                ]
                if tinybert
                else [
                    f"{label} in {corpus} {end_token}"
                    for corpus in self.args.task_list
                    for label in task_ner_labels[corpus]
                ]
            )
        else:
            label_text = (
                [
                    f"{label_tokenizer.cls_token} {label} {end_token}"
                    for corpus in self.args.task_list
                    for label in task_ner_labels[corpus]
                ]
                if tinybert
                else [
                    f"{label} {end_token}"
                    for corpus in self.args.task_list
                    for label in task_ner_labels[corpus]
                ]
            )
        tokenized_label = label_tokenizer(label_text, return_tensors="pt", padding=True)
        embeddings = label_encoder(**tokenized_label).last_hidden_state
        if self.args.label_emb_cls:
            labels_embedding = embeddings[:, 0, :]
        else:
            flag_cls = False
            labels_embedding = torch.zeros(self.num_ner_labels, embeddings.size(-1))
            for i, input_ids_list in enumerate(tokenized_label["input_ids"]):
                for j, input_ids in enumerate(input_ids_list):
                    if input_ids == label_tokenizer.cls_token_id:
                        flag_cls = True
                        continue
                    elif j == 0:
                        flag_cls = True
                    elif (
                        input_ids == end_token_id
                        or input_ids == label_tokenizer.convert_tokens_to_ids("in")
                    ):
                        labels_embedding[i, :] = label_embedding.mean(dim=0)
                        break
                    label_embedding = (
                        embeddings[i, j, :].unsqueeze(0)
                        if flag_cls
                        else torch.cat(
                            (label_embedding, embeddings[i, j, :].unsqueeze(0)),
                            dim=0,
                        )
                    )
                    flag_cls = False
        return labels_embedding.to(self.args.model_device).detach()

    def forward(
        self,
        input_ids,
        spans,
        spans_mask,
        spans_ner_label=None,
        token_type_ids=None,
        attention_mask=None,
        labels_mask=None,
        training=True,
        corpus_info=None,
    ):
        self.training = training
        spans_embedding = self.encode_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )  # , label_tensor # [batch_size, span_num, 768*2+150]

        # classification
        spans_embedding = self.dropout(self.relu(self.spans_embedding_layer(spans_embedding)))
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]
        logits = torch.mul(logits, labels_mask)

        # entity contrastive constraints
        if training and self.args.entity_const:
            entity_const_loss_tensor = torch.zeros(labels_mask.shape[1]).to(self.args.model_device)
            bool_other = True
            for i, mask in enumerate(labels_mask[0]):
                if mask == 1:
                    if bool_other:
                        bool_other = False
                        continue
                    label_spans_embedding = spans_embedding[spans_ner_label == i]
                    if label_spans_embedding.shape[0] == 0 or label_spans_embedding.shape[0] == 1:
                        continue
                    entity_const_var = torch.var(label_spans_embedding, dim=0) + 1e-8
                    entity_const_loss_tensor[i] += torch.mean(entity_const_var)
            entity_const_loss = torch.where(
                entity_const_loss_tensor > self.args.entity_const_th,
                entity_const_loss_tensor,
                0,
            )
            entity_const_loss = torch.sum(entity_const_loss)
        else:
            entity_const_loss = torch.tensor(0).to(self.args.model_device)

        if not self.args.simple and self.args.do_train and self.args.case_tsne is None:
            if self.args.general_one_hot == "other":
                one_hot_label = self.one_hot_label[spans_ner_label]
            elif (
                self.args.lookup_label_token
                or self.args.lookup_label_token_onehot is not None
                or self.args.saved_label_embedding_dir is not None
            ):
                one_hot_label = self.label_embedding(spans_ner_label)
            elif self.args.condition_dynamic_mu:
                one_hot_label = F.relu(self.dynamic_mu_layer(self.dynamic_mu))
                one_hot_label = one_hot_label[spans_ner_label]
            vae_enc_input = (
                spans_embedding
                if self.args.vae
                else torch.cat((spans_embedding, one_hot_label), dim=2)
            )
            spans_exp = F.relu(self.span_layer(vae_enc_input))
            mu = self.mu(spans_exp)
            logvar = self.log_var(spans_exp)
            z = self.reparameterize(mu, logvar)
            if not self.args.vae:
                z = torch.cat((z, one_hot_label), dim=2)
            pred_recon = self.decorde_span_embeddings(z)
            if self.args.dynamic_mu:
                cvae_loss = self.dynamic_cvae_loss_function(
                    spans_embedding, pred_recon, mu, logvar, spans_ner_label
                )
            else:
                cvae_loss = self.cvae_loss_function(spans_embedding, pred_recon, mu, logvar)
        else:
            mu = torch.tensor(0).to(self.args.model_device)
            logvar = torch.tensor(0).to(self.args.model_device)
            cvae_loss = torch.tensor(0).to(self.args.model_device)

        if training:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss,
                    spans_ner_label.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label),
                )
                loss = loss_fct(active_logits, active_labels)
                classification_loss = loss
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1))
                classification_loss = loss
            if self.args.entity_const:
                loss += self.args.entity_const_weight * entity_const_loss
            if not self.args.simple:
                loss += self.args.alpha * cvae_loss
                return (
                    loss,
                    logits,
                    spans_embedding,
                    mu,
                    logvar,
                    classification_loss,
                    cvae_loss,
                    entity_const_loss,
                )
            return loss, logits, spans_embedding
        else:
            if not self.args.simple and self.args.case_tsne is None:
                return logits, spans_embedding, mu, cvae_loss
            return logits, spans_embedding
