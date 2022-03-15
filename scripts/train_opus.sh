#!/bin/bash

source ~/.bashrc
conda activate mmt
ulimit -n 2048
# many to one
# SAVE_DIR=../checkpoints/opus/base/many-to-one/
# fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 5 --encoder-langtok src \
# --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
# --lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 32 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/

# SAVE_DIR=../checkpoints/opus_MoE/many-to-one/
# fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 5 --encoder-langtok tgt \
# --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
# --lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 16 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 16


# SAVE_DIR=../checkpoints/opus_suppressor_encoder/many-to-one/
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 5 --encoder-langtok src \
# --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
# --lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 50000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 512 --update-freq 64 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 1 --expert_type proj --switcher-proj 1 --switcher-fc 0


# SAVE_DIR=../checkpoints/opus_base_5_ar/many-to-one/
# fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 5 --encoder-langtok src \
# --langs de,nl,fr,pt,ro,es,en,ar \
# --lang-pairs  de-en,nl-en,fr-en,pt-en,ro-en,es-en,ar-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --max-update 12000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 16 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --iterative-normalization 0 --expert_num 1 --expert_type proj --switcher-proj 0 --switcher-fc 0

# Many to one
# hidden_size=$1
# name=${2}
# encoder=${3}
# decoder=${4}
# SAVE_DIR=../checkpoints/opus/small_LS_${hidden_size}_${name}/many-to-one/
# fairseq-train  ../data/opus-100/rebuilt-small/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 2 --encoder-langtok src \
# --langs de,es,it,nl,pl,ar,fa,he,en \
# --lang-pairs  de-en,es-en,it-en,nl-en,pl-en,ar-en,fa-en,he-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 4 --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234  --fp16  --fp16-init-scale 16 --ddp-backend no_c10d --patience 10 \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 0 --switcher-fc 1 --switcher-encoder ${encoder} --switcher-decoder ${decoder} --switcher-hidden-size ${hidden_size}

##### Single language 2gpus
# lg=$1
# SAVE_DIR=../checkpoints/opus/small_base_${lg}_3/many-to-one/
# fairseq-train  ../data/opus-100/rebuilt-small/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 1 --encoder-langtok src \
# --langs ${lg},en \
# --lang-pairs ${lg}-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --max-update 20000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 2 --save-interval-updates 200 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234  --fp16  --fp16-init-scale 16 --ddp-backend no_c10d \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 1 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 256



# one to many

hidden_size=$1
name=${2}
encoder=${3}
decoder=${4}
SAVE_DIR=../checkpoints/opus/small_LS_${hidden_size}_${name}/one-to-many/
fairseq-train  ../data/opus-100/rebuilt-small/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
--sampling-method temperature --sampling-temperature 2 --encoder-langtok tgt \
--langs de,es,it,nl,pl,ar,fa,he,en \
--lang-pairs   en-de,en-es,en-it,en-nl,en-pl,en-ar,en-fa,en-he \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 2000 --max-update 22000 --dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.0 --max-tokens 4096 --update-freq 4 --save-interval-updates 300 --keep-interval-updates 2 --no-epoch-checkpoints \
--log-format simple --log-interval 100 --seed 1234  --fp16  --fp16-init-scale 16 --ddp-backend no_c10d --patience 10 \
--save-dir ${SAVE_DIR} --max-source-positions 256 \
--max-target-positions 256 --skip-invalid-size-inputs-valid-test \
--tensorboard-logdir ${SAVE_DIR}/log/ \
--switcher-proj 0 --switcher-fc 1 --switcher-encoder ${encoder} --switcher-decoder ${decoder} --switcher-hidden-size ${hidden_size}

# 2GPUs
# lg=$1
# SAVE_DIR=../checkpoints/opus/small_base_${lg}/one-to-many/
# fairseq-train  ../data/opus-100/rebuilt-small/data-bin/ --arch transformer_iwslt_de_en_IN --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 1 --encoder-langtok tgt \
# --langs ${lg},en \
# --lang-pairs en-${lg} \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 400 --max-update 4000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 8 --save-interval-updates 200 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234  --fp16  --fp16-init-scale 16 --ddp-backend no_c10d \
# --save-dir ${SAVE_DIR} --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir ${SAVE_DIR}/log/ \
# --switcher-proj 1 --switcher-fc 1 --switcher-encoder 0 --switcher-decoder 0 --switcher-hidden-size 0
