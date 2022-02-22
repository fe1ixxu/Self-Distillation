#!/bin/bash

source ~/.bashrc
conda activate mmt
# many to one
# CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big --task translation_multi_simple_epoch \
# --sampling-method temperature --sampling-temperature 5 --encoder-langtok tgt \
# --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
# --lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
# --weight-decay 0.0 --max-tokens 4096 --update-freq 32 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
# --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
# --save-dir /brtx/604-nvme1/haoranxu/MMT/checkpoints/opus_base/many-to-one/ --max-source-positions 256 \
# --max-target-positions 256 --skip-invalid-size-inputs-valid-test \
# --tensorboard-logdir /brtx/604-nvme1/haoranxu/MMT/checkpoints/opus_base/many-to-one/log/

SAVE_DIR=../checkpoints/opus_MoE/many-to-one/
fairseq-train  ../data/opus-100/rebuilt/data-bin/ --arch transformer_vaswani_wmt_en_de_big_IN --task translation_multi_simple_epoch \
--sampling-method temperature --sampling-temperature 5 --encoder-langtok tgt \
--langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
--lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 25000 --dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.0 --max-tokens 4096 --update-freq 16 --save-interval-updates 500 --keep-interval-updates 2 --no-epoch-checkpoints \
--log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d --patience 5 \
--save-dir ${SAVE_DIR} --max-source-positions 256 \
--max-target-positions 256 --skip-invalid-size-inputs-valid-test \
--tensorboard-logdir ${SAVE_DIR}/log/ \
--iterative-normalization 0 --expert_num 16