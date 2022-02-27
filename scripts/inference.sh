MODEL_PATH=$1

# Many to One
DATA_DIR=../data/iwslt14/
for SRC in ar  de  es  fa  he  it  nl  pl  pt  ro  ru  sl  tr  zh; do
    TGT=en
    FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
    FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
    FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
    mkdir -p ${MODEL_PATH}/results

    cat $FSRC | python scripts/truncate.py | \
    CUDA_VISIBLE_DEVICES=0 fairseq-interactive ${DATA_DIR}/data-bin \
        --task translation_multi_simple_epoch --encoder-langtok src --path $MODEL_PATH/checkpoint_best.pt \
        --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
        --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
        --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece | \
    grep -P "^H" | cut -f 3- > $FOUT

    cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

done

# python scripts/get_mean_bleu.py --path ../checkpoints/iwslt_base/many-to-one/results/ --data iwslt

# One to Many 
# DATA_DIR=../data/iwslt14/
# for TGT in ar  de  es  fa  he  it  nl  pl  pt  ro  ru  sl  tr  zh; do
#     SRC=en
#     FSRC=${DATA_DIR}/tok/test.${TGT}-${SRC}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${TGT}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
#         --lang-pairs en-ar,en-es,en-he,en-nl,en-pt,en-ru,en-tr,en-de,en-fa,en-it,en-pl,en-ro,en-sl,en-zh \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done

# python scripts/get_mean_bleu.py --path ../checkpoints/iwslt_base/one-to-many/results/ --data iwslt --nen



## Latin to One
# DATA_DIR=../data/iwslt14/
# for SRC in de es it nl pl pt ro sl tr ar; do
#     TGT=en
#     FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     CUDA_VISIBLE_DEVICES=2 fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs de,es,it,nl,pl,pt,ro,sl,tr,en,ar \
#         --lang-pairs de-en,es-en,it-en,nl-en,pl-en,pt-en,ro-en,sl-en,tr-en,ar-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu
# done

# De -> One
# DATA_DIR=../data/iwslt14/
# for SRC in de ; do
#     TGT=en
#     FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     CUDA_VISIBLE_DEVICES=3 fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs de,en \
#         --lang-pairs de-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done

# IN Many to One
# DATA_DIR=../data/iwslt14/
# for SRC in ar  de  es  fa  he  it  nl  pl  pt  ro  ru  sl  tr  zh; do
#     TGT=en
#     FSRC=${DATA_DIR}/tok/test.${SRC}-${TGT}.${SRC}
#     FTGT=${DATA_DIR}/preprocessed/${SRC}/test.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${SRC}-${TGT}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     CUDA_VISIBLE_DEVICES=2 fairseq-interactive ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs ar,es,en,he,nl,pt,ru,tr,de,fa,it,pl,ro,sl,zh \
#         --lang-pairs ar-en,es-en,he-en,nl-en,pt-en,ru-en,tr-en,de-en,fa-en,it-en,pl-en,ro-en,sl-en,zh-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 256 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done


## OPUS many to one:
# DATA_DIR=../data/opus-100/rebuilt/
# for SRC in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
#     TGT=en
#     FSRC=${DATA_DIR}/test.${TGT}-${SRC}.${SRC}
#     FTGT=${DATA_DIR}/raw/test.${TGT}-${SRC}.${TGT}
#     FOUT=${MODEL_PATH}/results/test.${TGT}-${SRC}.${TGT}
#     mkdir -p ${MODEL_PATH}/results

#     cat $FSRC | python scripts/truncate.py | \
#     python fairseq_cli/interactive.py ${DATA_DIR}/data-bin \
#         --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH/checkpoint_best.pt \
#         --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
#         --lang-pairs es-en,fr-en,ro-en,nl-en,cs-en,el-en,hu-en,pl-en,tr-en,pt-en,bg-en,it-en,fi-en,hr-en,ar-en,sr-en,he-en,de-en,sl-en,ru-en,sv-en,da-en,et-en,bs-en,sk-en,id-en,no-en,fa-en,lt-en,zh-en,lv-en,mk-en,vi-en,th-en,ja-en,sq-en,ms-en,is-en,ko-en,uk-en,ca-en,eu-en,mt-en,gl-en,ml-en,bn-en,pa-en,hi-en,ta-en,si-en,nb-en,nn-en,te-en,gu-en,mr-en,ne-en,kn-en,or-en,as-en,ka-en,be-en,eo-en,cy-en,ga-en,ug-en,az-en,xh-en,af-en,oc-en,br-en,rw-en,km-en,ku-en,wa-en,mg-en,kk-en,tg-en,am-en,ps-en,my-en,uz-en,ur-en,ky-en,gd-en,sh-en,li-en,zu-en,fy-en,tk-en,yi-en,tt-en,se-en,ha-en,ig-en \
#         --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
#         --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar | \
#     grep -P "^H" | cut -f 3- > $FOUT

#     cat ${FOUT} | sacrebleu $FTGT -m bleu -b -w 2 > ${FOUT}.bleu

# done