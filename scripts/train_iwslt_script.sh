#!/bin/bash

source ~/.bashrc
conda activate mmt

## All Latin
fairseq-train ../data/iwslt14/data-bin/ --arch transformer_iwslt_de_en --task translation_multi_simple_epoch \
--sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
--langs de,es,it,nl,pl,pt,ro,sl,tr,en,ar \
--lang-pairs de-en,es-en,it-en,nl-en,pl-en,pt-en,ro-en,sl-en,tr-en,ar-en \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.0 --max-tokens 8192 --update-freq 8 --patience 10 \
--save-interval-updates 100 --keep-interval-updates 2 --no-epoch-checkpoints --log-format simple --log-interval 100 \
--seed 1234 --fp16 --ddp-backend no_c10d \
--save-dir ../checkpoints/iwslt_base/latin+ar-to-one/ --max-source-positions 256 --max-target-positions 256 \
--skip-invalid-size-inputs-valid-test --tensorboard-logdir ../checkpoints/iwslt_base/latin+ar-to-one/log/

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

