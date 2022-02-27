#!/bin/bash

source ~/.bashrc
conda activate mmt
# many to one
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
# --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 8192 --update-freq 8 --patience 10 \
# --save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --seed 1234 --fp16 --ddp-backend no_c10d \
# --save-dir ../checkpoints/iwslt_base/many-to-one/ --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ../checkpoints/iwslt_base/many-to-one/log/


# one to many
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
# --lang-pairs en-ar,en-es,en-he,en-nl,en-pt,en-ru,en-tr,en-de,en-fa,en-it,en-pl,en-ro,en-sl,en-zh \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 8192 --update-freq 8 --patience 10 \
# --save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --seed 1234 --fp16 --ddp-backend no_c10d \
# --save-dir ../checkpoints/iwslt_base/one-to-many/ --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ../checkpoints/iwslt_base/one-to-many/log/

# many to one with IN, MoE:
# SAVE_DIR=../checkpoints/iwslt_MoE_ffn/many-to-one/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
# --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 32 --patience 10 \
# --save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 500 \
# --seed 1234 --fp16 --ddp-backend no_c10d \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 4 --expert_type ffn

# SAVE_DIR=../checkpoints/iwslt_MoE_ffn/many-to-one/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
# --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --patience 5 \
# --save-interval-updates 1000 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 500 \
# --ddp-backend no_c10d \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 4 --expert_type ffn

# SAVE_DIR=../checkpoints/iwslt_switcher_proj_sigmoid_encoder_only/many-to-one/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok src \
# --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
# --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0004 --warmup-updates 2000 --max-update 60000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 512 --update-freq 32 --patience 5 \
# --save-interval-updates 1000 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 500 \
# --ddp-backend no_c10d --fp16  \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 1 --expert_type proj --switcher 1


SAVE_DIR=../checkpoints/iwslt_switcher_proj_ffc_sigmoid_encoder_only/many-to-one/
fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
--sampling-method temperature --sampling-temperature 2 --encoder-langtok src \
--langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
--lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 0.0004 --warmup-updates 2000 --max-update 60000 --dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.0 --max-tokens 1024 --update-freq 16 --patience 5 \
--save-interval-updates 1000 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 500 \
--ddp-backend no_c10d --fp16  \
--save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
--skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
--iterative-normalization 0 --expert_num 1 --expert_type proj --switcher-proj 1 --switcher-fc 1
