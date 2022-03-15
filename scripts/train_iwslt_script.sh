#!/bin/bash

source ~/.bashrc
conda activate mmt

## All Latin
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs de,es,it,nl,pl,pt,ro,sl,tr,en,ar \
# --lang-pairs de-en,es-en,it-en,nl-en,pl-en,pt-en,ro-en,sl-en,tr-en,ar-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 8192 --update-freq 8 --patience 10 \
# --save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --seed 1234 --fp16 --ddp-backend no_c10d \
# --save-dir ../checkpoints/iwslt_base/latin+ar-to-one/ --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ../checkpoints/iwslt_base/latin+ar-to-one/log/

## De -> En
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs de,en \
# --lang-pairs de-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 8192 --update-freq 8 --patience 10 \
# --save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --seed 1234 --fp16 --ddp-backend no_c10d \
# --save-dir ../checkpoints/iwslt_base/de-to-one/ --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ../checkpoints/iwslt_base/de-to-one/log/


#Many to one 5 langs to En
# SAVE_DIR=../checkpoints/iwslt_new2/base_5/many-to-one/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok src \
# --langs de,es,it,nl,pl,en \
# --lang-pairs de-en,es-en,it-en,nl-en,pl-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1600 --max-update 16000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --patience 3 \
# --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --ddp-backend no_c10d --fp16  --fp16-init-scale 16 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 1 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 0

#Many to one 3 langs to En
# SAVE_DIR=../checkpoints/iwslt_new2/base_3/many-to-one/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok src \
# --langs ar,fa,he,en \
# --lang-pairs ar-en,fa-en,he-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1200 --max-update 12000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --patience 3 \
# --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --ddp-backend no_c10d --fp16  --fp16-init-scale 16 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 1 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 0

# One to Many 5 langs to x
# SAVE_DIR=../checkpoints/iwslt_new2/base_5/one-to-many/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs de,es,it,nl,pl,en \
# --lang-pairs en-de,en-es,en-it,en-nl,en-pl \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1600 --max-update 16000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --patience 3 \
# --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --ddp-backend no_c10d --fp16  --fp16-init-scale 16 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 0 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 0

# One to Many 3 langs to x
# SAVE_DIR=../checkpoints/iwslt_new2/base_3/one-to-many/
# fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
# --langs ar,fa,he,en \
# --lang-pairs en-ar,en-fa,en-he \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1200 --max-update 12000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --patience 3 \
# --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
# --ddp-backend no_c10d --fp16  --fp16-init-scale 16 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 --max-target-positions 256 \
# --skip-invalid-size-inputs-valid-test --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 0 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 0

